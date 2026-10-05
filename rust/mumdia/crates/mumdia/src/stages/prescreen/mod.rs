//! Fragment-rarity candidate prescreen `mumdia prescreen`: score every library candidate on the
//! spectra of its own isolation window and calibrated retention-time range, and keep the
//! candidates whose score exceeds a label-blind calibration quantile.
//!
//! Score. For each eligible spectrum (the spectrum's isolation window holds the precursor m/z,
//! `lower <= m < upper`, and its retention time lies inside the candidate's `run_windows`
//! bounds) and each orientation of the residue array (forward, and fully reversed when
//! `both_orientations`), the b and y fragments `k = 1..L-1` at charges `1..=min(max_frag_charge,
//! z)` are matched to the nearest peak within `frag_tol_da`. A matched peak counts once however
//! many fragments land on it and only when its intensity is positive. Each counted peak adds
//! `-ln p / sqrt(L)`, times `low_mz_weight` below `low_mz_threshold`, where `p` is the share of
//! the window's spectra holding a peak in the 0.01 Da bins `b-1..=b+1` around it (`b =
//! rint(100 m/z)`, a bin counted once per spectrum, `p = min(0.99999, (count + 0.5) / (n + 1))`).
//! The candidate's score is the maximum over spectra and orientations. The histogram reads the
//! spectra only; no identification, label or library enters it.
//!
//! Calibration. The in-scope candidates are split into a calibration and a reporting half by a
//! seeded hash of the candidate id. The cutoff is the `target` quantile of the calibration
//! half's scores by numpy's "higher" method (`sorted[ceil((n - 1) q)]`), and a candidate is kept
//! when its score is strictly greater. Out-of-scope candidates pass through. The reporting half
//! gives the reduction estimate that was not used to set the cutoff.
//!
//! Exchangeability. Targets and decoys are scored by the identical rule, each on its own residues,
//! m/z and RT window, and the label is read only for the survivors table and the report. With
//! `both_orientations` a candidate's score is invariant under reversal of its residue array (the
//! reversed orientation of a reversed decoy is its target's forward orientation), which is the
//! property the measured prototype checked. Treat the stage as a compute reduction.
//!
//! Output. The survivors use the `prescan_survivors` contract (`candidate_id`, `label`, sorted),
//! so `extract --restrict-candidates` consumes either stage's table.
//!
//! Provenance: a port of the `tagbench` prototype's `advanced_candidates.scores` (the
//! `lowmz_half` column) and `advanced_filter.export` (docs/34_prescreen.md).

mod evidence;
mod ext;
pub mod masses;
pub mod retrieval;
pub mod tags;
pub mod traces;

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{
    Config, PrescreenConfig, PrescreenLocalization, PrescreenRetrieval, PrescreenScope,
    PrescreenScopeMatch,
};
use mumdia_core::constants::{residue_mass, PROTON, WATER};
use mumdia_core::mass::{parse_peptidoform, unimod_mass};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, TableFile};
use rayon::prelude::*;
use serde_json::json;
use tracing::{info, warn};

pub struct PrescreenParams<'a> {
    /// `spectra_ms2` for this run.
    pub ms2: &'a str,
    /// Library precursors: `candidate_id, peptidoform, charge, precursor_mz, label`.
    pub library_precursors: &'a str,
    /// Per-candidate RT bounds (`candidate_id, rt_lo, rt_hi`). `None` screens every candidate
    /// over the whole gradient.
    pub run_windows: Option<&'a str>,
    /// `spectra_ms1` for this run; needed only by `prescreen.mass_hypotheses.ms1`.
    pub ms1: Option<&'a str>,
    pub out: &'a str,
    /// The whole configuration: the tag alphabet comes from its chemistry block.
    pub config: &'a Config,
    pub config_hash: &'a str,
    /// Evaluation only: score a seeded, label-blind sample of about this many candidates and
    /// nothing else (0 = every candidate). The survivors table then covers the sample only and
    /// must not be handed to extract; it exists to measure expensive components the way the
    /// prototype did, on sampled candidates.
    pub sample_candidates: usize,
    /// Tag prefilter: keep the candidates the database-free tags retrieve and compute no
    /// fragment score; the calibration, presets and optional components do not apply.
    pub tags_only: bool,
}

/// What the stage kept, for the orchestrator's log and manifest.
#[derive(Clone, Debug, Default)]
pub struct PrescreenSummary {
    pub screened: u64,
    pub kept: u64,
    pub cutoff: Option<f64>,
    pub bypass_reason: Option<String>,
}

/// Fragment-rarity bin of an m/z: `rint(100 m/z)`, ties to even as numpy's `rint`.
#[inline]
fn bin_of(mz: f64) -> i64 {
    (mz * 100.0).round_ties_even() as i64
}

/// Per-window spectrum-presence histogram over 0.01 Da bins.
struct Histogram {
    counts: Vec<u32>,
    n_spectra: u32,
}

impl Histogram {
    /// Count, per bin, the spectra holding at least one peak in it. `peaks` yields each
    /// spectrum's m/z values.
    fn build<'a>(spectra: impl Iterator<Item = &'a [f32]>, nbin: usize) -> Histogram {
        let mut counts = vec![0u32; nbin];
        let mut stamp = vec![u32::MAX; nbin];
        let mut n = 0u32;
        for mz in spectra {
            for &m in mz {
                let b = bin_of(m as f64);
                if b >= 0 && (b as usize) < nbin && stamp[b as usize] != n {
                    counts[b as usize] += 1;
                    stamp[b as usize] = n;
                }
            }
            n += 1;
        }
        Histogram {
            counts,
            n_spectra: n,
        }
    }

    /// `min(0.99999, (sum of bins b-1..=b+1 + 0.5) / (n + 1))`.
    fn probability(&self, mz: f64) -> f64 {
        self.probability_bin(bin_of(mz))
    }

    /// `probability` of any m/z in bin `b`: the weight depends on the bin alone.
    fn probability_bin(&self, b: i64) -> f64 {
        let mut c = 0u64;
        for k in b - 1..=b + 1 {
            if k >= 0 && (k as usize) < self.counts.len() {
                c += self.counts[k as usize] as u64;
            }
        }
        ((c as f64 + 0.5) / (self.n_spectra as f64 + 1.0)).min(0.99999)
    }
}

/// A window's peak-occupancy bitmap: one bit per 0.01 Da bin (`bin_of`) per spectrum, set when the
/// spectrum has any peak there. A peak within `tol` of `x` lies in the bins `bin_of(x - tol)` to
/// `bin_of(x + tol)`; the range is widened by one bin on each side so that rounding can never drop
/// a real match. A fragment whose range is empty in a spectrum cannot match it, so the binary
/// search is skipped: scores are identical, and most searches in dense data are skipped.
struct Occupancy {
    words: usize,
    bits: Vec<u64>,
    /// Inverted index: the window's spectra (local index, ascending) holding a peak in bin `b`
    /// are `post[post_off[b]..post_off[b + 1]]`.
    post_off: Vec<u32>,
    post: Vec<u32>,
    /// `-ln p` of bin `b`, the rarity weight every peak in that bin carries.
    wbin: Vec<f64>,
}

impl Occupancy {
    fn build(sp: &Spectra, g: &WindowGroup, hist: &Histogram) -> Occupancy {
        let words = sp.nbin / 64 + 2;
        let nbits = words * 64;
        let mut bits = vec![0u64; g.count * words];
        bits.par_chunks_mut(words).enumerate().for_each(|(t, row)| {
            for &m in sp.peaks(g.first + t) {
                let b = bin_of(m as f64);
                if b >= 0 && (b as usize) < nbits {
                    row[b as usize / 64] |= 1u64 << (b as usize % 64);
                }
            }
        });
        // The inverted index from the bitmap rows, spectra ascending within each bin.
        let mut counts = vec![0u32; nbits + 1];
        for row in bits.chunks(words) {
            for (w, &word) in row.iter().enumerate() {
                let mut x = word;
                while x != 0 {
                    counts[w * 64 + x.trailing_zeros() as usize] += 1;
                    x &= x - 1;
                }
            }
        }
        let mut post_off = vec![0u32; nbits + 1];
        for b in 0..nbits {
            post_off[b + 1] = post_off[b] + counts[b];
        }
        let mut fill = post_off.clone();
        let mut post = vec![0u32; post_off[nbits] as usize];
        for (t, row) in bits.chunks(words).enumerate() {
            for (w, &word) in row.iter().enumerate() {
                let mut x = word;
                while x != 0 {
                    let b = w * 64 + x.trailing_zeros() as usize;
                    post[fill[b] as usize] = t as u32;
                    fill[b] += 1;
                    x &= x - 1;
                }
            }
        }
        let wbin = (0..nbits as i64)
            .map(|b| -hist.probability_bin(b).ln())
            .collect();
        Occupancy {
            words,
            bits,
            post_off,
            post,
            wbin,
        }
    }

    /// The window spectra (local index) with a peak in bin `b`, restricted to `lo..hi`.
    fn spectra_in(&self, b: i64, lo: usize, hi: usize) -> &[u32] {
        if b < 0 || b as usize + 1 >= self.post_off.len() {
            return &[];
        }
        let all =
            &self.post[self.post_off[b as usize] as usize..self.post_off[b as usize + 1] as usize];
        let a = all.partition_point(|&t| (t as usize) < lo);
        let z = all.partition_point(|&t| (t as usize) < hi);
        &all[a..z]
    }

    fn row(&self, t: usize) -> &[u64] {
        &self.bits[t * self.words..(t + 1) * self.words]
    }
}

/// One spectrum's occupancy bits and the fragments' bin ranges.
type OccupancyRow<'a> = (&'a [u64], &'a [(i64, i64)]);

/// Any bit set in `row` over the inclusive bin range.
#[inline]
fn any_bit(row: &[u64], lo: i64, hi: i64) -> bool {
    let lo = lo.max(0) as usize;
    let hi = (hi.max(0) as usize).min(row.len() * 64 - 1);
    (lo..=hi).any(|b| row[b / 64] >> (b % 64) & 1 == 1)
}

/// The widened bin range a fragment's matches can lie in.
#[inline]
fn fragment_bins(x: f64, tol: f64) -> (i64, i64) {
    (bin_of(x - tol) - 1, bin_of(x + tol) + 1)
}

/// Index of the peak nearest `x` within `tol` in ascending `mz`, the first on a tie; the
/// prototype's `nearest_index`. Comparisons are in f64 on the widened f32 values.
#[inline]
fn nearest_index(mz: &[f32], x: f64, tol: f64) -> Option<usize> {
    let lo = x - tol;
    let hi = x + tol;
    let mut j = mz.partition_point(|&m| (m as f64) < lo);
    let mut best = None;
    let mut err = tol + 1.0;
    while j < mz.len() && (mz[j] as f64) <= hi {
        let d = (mz[j] as f64 - x).abs();
        if d < err {
            best = Some(j);
            err = d;
        }
        j += 1;
    }
    best
}

/// Theoretical b and y m/z values for one orientation of a residue-mass array, `k = 1..L-1`,
/// charges `1..=nz`.
fn fragment_mz(masses: &[f64], reversed: bool, nz: i32, out: &mut Vec<f64>) {
    // The prototype's order: b at charge 1 (k = 1..L-1), y at charge 1, then charge 2. The
    // score sums distinct peaks, so only the repeat bonus's fragment ownership sees the order.
    out.clear();
    let l = masses.len();
    let at = |i: usize| {
        if reversed {
            masses[l - 1 - i]
        } else {
            masses[i]
        }
    };
    let nc = l.saturating_sub(1);
    let mut bm = vec![0.0f64; nc];
    let mut ym = vec![0.0f64; nc];
    let (mut b, mut y) = (0.0, 0.0);
    for k in 0..nc {
        b += at(k);
        y += at(l - 1 - k);
        bm[k] = b;
        ym[k] = y;
    }
    for z in 1..=nz {
        let zf = z as f64;
        out.extend(bm.iter().map(|&m| m / zf + PROTON));
        out.extend(ym.iter().map(|&m| (m + WATER) / zf + PROTON));
    }
}

/// Residue masses with their modifications, the N-terminal delta folded into the first residue
/// and the C-terminal delta into the last, so a reversal moves each with its residue.
fn residue_masses(pf: &str) -> Option<Vec<f64>> {
    let p = parse_peptidoform(pf).ok()?;
    let mut m: Vec<f64> = p
        .residues
        .iter()
        .zip(&p.mods)
        .map(|(&r, &d)| residue_mass(r).map(|x| x + d))
        .collect::<Option<_>>()?;
    let l = m.len();
    if l == 0 {
        return None;
    }
    m[0] += p.n_term_mod;
    m[l - 1] += p.c_term_mod;
    Some(m)
}

/// One `RESIDUE:Name` scope entry resolved to a residue (or terminus) and a mass delta.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ScopeSite {
    Residue(u8),
    NTerm,
    CTerm,
}

#[derive(Clone, Debug)]
pub(crate) struct ScopeMod {
    site: ScopeSite,
    delta: f64,
}

/// Mass agreement for "this residue carries that modification": the peptidoform parser sums a
/// residue's deltas, and named, `UNIMOD:` and numeric spellings all resolve to the same mass.
const SCOPE_MASS_TOL: f64 = 1e-3;

pub(crate) fn resolve_scope(specs: &[String]) -> Result<Vec<ScopeMod>> {
    specs
        .iter()
        .map(|s| {
            let (r, name) = s.split_once(':').ok_or_else(|| {
                anyhow::anyhow!("prescreen.scope_mods entry '{s}' must be RESIDUE:Name")
            })?;
            let delta = unimod_mass(name)
                .or_else(|| name.parse::<f64>().ok())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "prescreen.scope_mods names an unknown modification '{name}'; add it to \
                         the shared mass model or give the delta as a number"
                    )
                })?;
            let site = match r {
                "n" => ScopeSite::NTerm,
                "c" => ScopeSite::CTerm,
                _ => ScopeSite::Residue(r.as_bytes()[0].to_ascii_uppercase()),
            };
            Ok(ScopeMod { site, delta })
        })
        .collect()
}

/// Whether a peptidoform falls in `scope = "modified"`.
pub(crate) fn in_scope(pf: &str, mods: &[ScopeMod], how: PrescreenScopeMatch) -> Option<bool> {
    let p = parse_peptidoform(pf).ok()?;
    let carries = |m: &ScopeMod| match m.site {
        ScopeSite::NTerm => (p.n_term_mod - m.delta).abs() < SCOPE_MASS_TOL,
        ScopeSite::CTerm => (p.c_term_mod - m.delta).abs() < SCOPE_MASS_TOL,
        ScopeSite::Residue(r) => p
            .residues
            .iter()
            .zip(&p.mods)
            .any(|(&x, &d)| x == r && (d - m.delta).abs() < SCOPE_MASS_TOL),
    };
    Some(match how {
        PrescreenScopeMatch::Any => mods.iter().any(carries),
        PrescreenScopeMatch::All => mods.iter().all(carries),
    })
}

/// Seeded calibration/reporting split: SplitMix64 of `seed ^ candidate_id`, low bit 0 =
/// calibration. Independent of row order and thread count.
#[inline]
pub(crate) fn is_calibration(seed: u64, cid: u32) -> bool {
    let mut z = seed ^ (cid as u64);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z & 1 == 0
}

/// numpy `quantile(x, q, method="higher")` on an already sorted slice.
pub(crate) fn quantile_higher(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let i = ((n - 1) as f64 * q).ceil() as usize;
    sorted[i.min(n - 1)]
}

/// Spectra of one isolation window (one distinct `(lower, upper)` pair), sorted by RT.
struct WindowGroup {
    lower: f64,
    upper: f64,
    /// Global spectrum indices into the reordered peak arrays, ascending RT.
    first: usize,
    count: usize,
}

/// The run's MS2 peaks reordered window by window and, inside a window, by retention time; each
/// spectrum's peaks ascending by m/z.
struct Spectra {
    rt: Vec<f64>,
    /// `offsets[s]..offsets[s + 1]` are spectrum `s`'s peaks.
    offsets: Vec<usize>,
    mz: Vec<f32>,
    /// Peak intensity > 0. A zero-intensity nearest peak is not a match, as in the prototype.
    positive: Vec<bool>,
    /// Peak intensities, kept only when the trace component needs them.
    intensity: Vec<f32>,
    groups: Vec<WindowGroup>,
    nbin: usize,
}

impl Spectra {
    fn load(path: &str, top_peaks: usize, keep_intensity: bool) -> Result<Spectra> {
        let t = TableFile::open(path)?;
        let rt = t.f64("rt_seconds")?;
        let lo = t.f64("window_lower")?;
        let hi = t.f64("window_upper")?;
        let (moff, mzs) = t.list_f32_flat("mz")?;
        let (ioff, ints) = t.list_f32_flat("intensity")?;
        drop(t);
        anyhow::ensure!(
            moff == ioff,
            "spectra_ms2: mz and intensity lists differ in length"
        );
        // Window groups by the exact bound pair, as the prototype's `np.unique(scans[:, 1:3])`.
        let mut order: Vec<usize> = (0..rt.len())
            .filter(|&s| lo[s].is_finite() && hi[s].is_finite())
            .collect();
        order.sort_by(|&a, &b| {
            lo[a]
                .total_cmp(&lo[b])
                .then(hi[a].total_cmp(&hi[b]))
                .then(rt[a].total_cmp(&rt[b]))
                .then(a.cmp(&b))
        });
        let per: Vec<Vec<(f32, f32)>> = order
            .par_iter()
            .map(|&s| {
                let mut p: Vec<(f32, f32)> = mzs[moff[s]..moff[s + 1]]
                    .iter()
                    .copied()
                    .zip(ints[moff[s]..moff[s + 1]].iter().copied())
                    .filter(|(m, _)| m.is_finite() && *m >= 0.0)
                    .collect();
                if top_peaks > 0 && p.len() > top_peaks {
                    // The prescan's tie-break: intensity descending, then ascending m/z.
                    p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.total_cmp(&b.0)));
                    p.truncate(top_peaks);
                }
                p.sort_by(|a, b| a.0.total_cmp(&b.0));
                p
            })
            .collect();
        drop(mzs);
        drop(ints);
        let total: usize = per.iter().map(Vec::len).sum();
        let mut offsets = Vec::with_capacity(order.len() + 1);
        let mut mz = Vec::with_capacity(total);
        let mut positive = Vec::with_capacity(total);
        let mut intensity = Vec::with_capacity(if keep_intensity { total } else { 0 });
        let mut max_mz = 0.0f64;
        offsets.push(0);
        for p in &per {
            for &(m, i) in p {
                mz.push(m);
                positive.push(i > 0.0);
                if keep_intensity {
                    intensity.push(i);
                }
                max_mz = max_mz.max(m as f64);
            }
            offsets.push(mz.len());
        }
        drop(per);
        let mut groups: Vec<WindowGroup> = Vec::new();
        for (k, &s) in order.iter().enumerate() {
            match groups.last_mut() {
                Some(g) if g.lower == lo[s] && g.upper == hi[s] => g.count += 1,
                _ => groups.push(WindowGroup {
                    lower: lo[s],
                    upper: hi[s],
                    first: k,
                    count: 1,
                }),
            }
        }
        let rt = order.iter().map(|&s| rt[s]).collect();
        Ok(Spectra {
            rt,
            offsets,
            mz,
            positive,
            intensity,
            groups,
            nbin: (max_mz * 100.0).ceil() as usize + 2,
        })
    }

    fn peaks(&self, s: usize) -> &[f32] {
        &self.mz[self.offsets[s]..self.offsets[s + 1]]
    }
}

/// Candidate inputs shared by every window. The residue masses are not stored: they follow
/// from the peptidoform and are computed when the candidate is scored, which keeps a
/// 200M-candidate library from holding 200M small heap vectors.
struct Candidate {
    nz: i32,
    /// Inclusive RT bounds, or `None` for the whole gradient.
    rt: Option<(f64, f64)>,
}

/// Score one candidate against the eligible spectra of one window group.
#[allow(clippy::too_many_arguments)]
fn score_in_group(
    c: &Candidate,
    masses: &[f64],
    sp: &Spectra,
    g: &WindowGroup,
    occ: Option<&Occupancy>,
    neglnp: &[f64],
    cfg: &PrescreenConfig,
    frag: &mut Vec<f64>,
    hits: &mut Vec<usize>,
    trace: &mut Vec<f64>,
) -> (f64, f64) {
    let rts = &sp.rt[g.first..g.first + g.count];
    let (a, b) = match c.rt {
        Some((lo, hi)) => (
            rts.partition_point(|&r| r < lo),
            rts.partition_point(|&r| r <= hi),
        ),
        None => (0, rts.len()),
    };
    if a >= b {
        return (0.0, 0.0);
    }
    let sqrt_l = (masses.len() as f64).sqrt();
    let base = sp.offsets[g.first];
    let mut best = 0.0f64;
    let mut best_repeat = 0.0f64;
    let repeat = cfg.repeat_bonus > 0.0;
    // Crowding: a spectrum's score divided by (its peaks / the window's mean)^exponent, the
    // ratio clamped to [0.25, 4] (phase 11 of the prototype).
    let mean_peaks = (sp.offsets[g.first + g.count] - base) as f64 / (g.count as f64).max(1.0);
    let orientations: &[bool] = if cfg.both_orientations {
        &[false, true]
    } else {
        &[false]
    };
    // Pruned exact path (no repeat bonus): an upper bound per spectrum from the inverted index,
    // then exact scores in descending bound order until no remaining bound can beat the best.
    if let (Some(o), false) = (occ, repeat) {
        let ne = b - a;
        let mut ub = vec![0.0f64; ne];
        let mut touched: Vec<usize> = Vec::new();
        for &rev in orientations {
            fragment_mz(masses, rev, c.nz, frag);
            let fbins: Vec<(i64, i64)> = frag
                .iter()
                .map(|&x| fragment_bins(x, cfg.frag_tol_da))
                .collect();
            for &(lo, hi) in &fbins {
                for bin in lo..=hi {
                    let w = match o.wbin.get(bin.max(0) as usize) {
                        Some(&w) if bin >= 0 => w,
                        _ => continue,
                    };
                    for &t in o.spectra_in(bin, a, b) {
                        let k = t as usize - a;
                        if ub[k] == 0.0 {
                            touched.push(k);
                        }
                        ub[k] += w;
                    }
                }
            }
            let crowd = |s: usize| {
                if cfg.crowding_exponent > 0.0 {
                    let n = (sp.offsets[s + 1] - sp.offsets[s]) as f64;
                    (n / mean_peaks.max(1e-12))
                        .clamp(0.25, 4.0)
                        .powf(cfg.crowding_exponent)
                } else {
                    1.0
                }
            };
            let mut order: Vec<(f64, usize)> = touched
                .iter()
                .map(|&k| (ub[k] / sqrt_l / crowd(g.first + a + k), k))
                .collect();
            order.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
            for &(bound, k) in &order {
                // A 1e-9 relative margin keeps rounding from pruning the true maximum.
                if bound * (1.0 + 1e-9) <= best {
                    break;
                }
                let s = g.first + a + k;
                let (lo, hi) = (sp.offsets[s], sp.offsets[s + 1]);
                let total = spectrum_score(
                    frag,
                    &sp.mz[lo..hi],
                    &sp.positive[lo..hi],
                    &neglnp[lo - base..hi - base],
                    sqrt_l,
                    cfg,
                    hits,
                    None,
                    Some((o.row(s - g.first), fbins.as_slice())),
                ) / crowd(s);
                best = best.max(total);
            }
            for &k in &touched {
                ub[k] = 0.0;
            }
            touched.clear();
        }
        return (best, 0.0);
    }
    for &rev in orientations {
        fragment_mz(masses, rev, c.nz, frag);
        let fbins: Vec<(i64, i64)> = frag
            .iter()
            .map(|&x| fragment_bins(x, cfg.frag_tol_da))
            .collect();
        let nf = frag.len();
        let ne = b - a;
        if repeat {
            trace.clear();
            trace.resize(ne * nf, 0.0);
        }
        for (t, s) in (g.first + a..g.first + b).enumerate() {
            let (lo, hi) = (sp.offsets[s], sp.offsets[s + 1]);
            let row = if repeat {
                Some(&mut trace[t * nf..(t + 1) * nf])
            } else {
                None
            };
            let mut total = spectrum_score(
                frag,
                &sp.mz[lo..hi],
                &sp.positive[lo..hi],
                &neglnp[lo - base..hi - base],
                sqrt_l,
                cfg,
                hits,
                row,
                occ.map(|o| (o.row(s - g.first), fbins.as_slice())),
            );
            if cfg.crowding_exponent > 0.0 {
                let ratio = ((hi - lo) as f64 / mean_peaks.max(1e-12)).clamp(0.25, 4.0);
                total /= ratio.powf(cfg.crowding_exponent);
            }
            best = best.max(total);
        }
        // Repeat support: each fragment counts as strongly as its weaker appearance between this
        // scan and its best neighbour within two eligible scans and 8 s (phase 11).
        if repeat {
            let rt_of = |t: usize| rts[a + t];
            for t in 0..ne {
                let mut sum = 0.0;
                for f in 0..nf {
                    let w = trace[t * nf + f];
                    if w <= 0.0 {
                        continue;
                    }
                    let mut nb = 0.0f64;
                    for v in t.saturating_sub(2)..(t + 3).min(ne) {
                        if v != t && (rt_of(v) - rt_of(t)).abs() <= 8.0 {
                            nb = nb.max(trace[v * nf + f]);
                        }
                    }
                    sum += w.min(nb);
                }
                best_repeat = best_repeat.max(sum / sqrt_l);
            }
        }
    }
    (best, best_repeat)
}

/// One spectrum's score for one fragment list: every peak that is the nearest positive-intensity
/// peak of at least one fragment counts once, in ascending peak order, as
/// `-ln p / sqrt(L)` times the low-m/z weight. With `owners`, the first fragment matching each
/// peak also records that peak's weight in its slot (the repeat bonus's per-fragment trace).
#[allow(clippy::too_many_arguments)]
fn spectrum_score(
    frag: &[f64],
    mz: &[f32],
    positive: &[bool],
    neglnp: &[f64],
    sqrt_l: f64,
    cfg: &PrescreenConfig,
    hits: &mut Vec<usize>,
    owners: Option<&mut [f64]>,
    occupancy: Option<OccupancyRow>,
) -> f64 {
    hits.clear();
    // Skip the search for a fragment with no peak anywhere in its bin range (exact).
    let present = |f: usize| match occupancy {
        Some((row, bins)) => any_bit(row, bins[f].0, bins[f].1),
        None => true,
    };
    if let Some(row) = owners {
        // The first fragment matching a peak owns it; its weight is the peak's rarity weight.
        for (f, &x) in frag.iter().enumerate() {
            if !present(f) {
                continue;
            }
            if let Some(p) = nearest_index(mz, x, cfg.frag_tol_da) {
                if positive[p] && !hits.contains(&p) {
                    hits.push(p);
                    let low = if (mz[p] as f64) < cfg.low_mz_threshold {
                        cfg.low_mz_weight
                    } else {
                        1.0
                    };
                    row[f] = neglnp[p] * low;
                }
            }
        }
        hits.clear();
    }
    for (f, &x) in frag.iter().enumerate() {
        if !present(f) {
            continue;
        }
        if let Some(p) = nearest_index(mz, x, cfg.frag_tol_da) {
            if positive[p] {
                hits.push(p);
            }
        }
    }
    hits.sort_unstable();
    hits.dedup();
    let mut total = 0.0;
    for &p in hits.iter() {
        let value = neglnp[p] / sqrt_l;
        let w = if (mz[p] as f64) < cfg.low_mz_threshold {
            cfg.low_mz_weight
        } else {
            1.0
        };
        total += value * w;
    }
    total
}

/// Candidate lists per window group: the groups whose bounds hold the precursor m/z.
fn group_members(sp: &Spectra, cands: &[Option<Candidate>], pmz: &[f64]) -> Vec<Vec<u32>> {
    let mut by_lower: Vec<(f64, f64, usize)> = sp
        .groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.lower, g.upper, i))
        .collect();
    by_lower.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); sp.groups.len()];
    for (i, c) in cands.iter().enumerate() {
        if c.is_none() {
            continue;
        }
        let m = pmz[i];
        let k = by_lower.partition_point(|w| w.0 <= m);
        for &(lo, hi, g) in &by_lower[..k] {
            if lo <= m && m < hi {
                members[g].push(i as u32);
            }
        }
    }
    members
}

/// Score every candidate: NaN for a candidate whose peptidoform cannot be parsed.
fn score_all(
    sp: &Spectra,
    cands: &[Option<Candidate>],
    members: &[Vec<u32>],
    cfg: &PrescreenConfig,
    masses_of: &(dyn Fn(usize) -> Vec<f64> + Sync),
) -> (Vec<f64>, Vec<f64>) {
    let mut repeat = vec![0.0f64; cands.len()];
    let mut score: Vec<f64> = cands
        .iter()
        .map(|c| if c.is_some() { 0.0 } else { f64::NAN })
        .collect();
    for (gi, g) in sp.groups.iter().enumerate() {
        if members[gi].is_empty() {
            continue;
        }
        let hist = Histogram::build((g.first..g.first + g.count).map(|s| sp.peaks(s)), sp.nbin);
        let base = sp.offsets[g.first];
        let end = sp.offsets[g.first + g.count];
        let occ = Occupancy::build(sp, g, &hist);
        let neglnp: Vec<f64> = sp.mz[base..end]
            .par_iter()
            .map(|&m| -hist.probability(m as f64).ln())
            .collect();
        let got: Vec<(f64, f64)> = members[gi]
            .par_iter()
            .map_init(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(frag, hits, trace), &i| {
                    let c = cands[i as usize].as_ref().expect("member is parseable");
                    let m = masses_of(i as usize);
                    score_in_group(c, &m, sp, g, Some(&occ), &neglnp, cfg, frag, hits, trace)
                },
            )
            .collect();
        for (&i, (v, r)) in members[gi].iter().zip(got) {
            let s = &mut score[i as usize];
            *s = s.max(v);
            let q = &mut repeat[i as usize];
            *q = q.max(r);
        }
    }
    (score, repeat)
}

/// The orchestrators' hook: when `prescreen.enabled`, screen `lib_p` on this run's MS2 inside
/// the fitted `windows` and return the survivors table for extract's `restrict_candidates`.
pub fn run_if_enabled(
    cfg: &mumdia_core::config::Config,
    config_hash: &str,
    ms2: &str,
    ms1: Option<&str>,
    lib_p: &str,
    windows: &str,
    out_dir: &str,
) -> Result<Option<String>> {
    if !cfg.prescreen.enabled {
        return Ok(None);
    }
    let out = format!("{out_dir}/prescreen_survivors.parquet");
    info!(stage = %"prescreen", "run: stage start");
    run(PrescreenParams {
        ms2,
        library_precursors: lib_p,
        run_windows: Some(windows),
        ms1,
        out: &out,
        config: cfg,
        config_hash,
        sample_candidates: 0,
        tags_only: false,
    })?;
    Ok(Some(out))
}

/// The orchestrator's hook before prediction, without any retention time: the database-free
/// tag prefilter (`prescreen.tag_prefilter`), the fragment-rarity score over the whole gradient
/// of each candidate's isolation window (`prescreen.score_before_prediction`, with the preset,
/// target and crowding settings of `prescreen`), or both, a candidate then having to pass both.
/// `table` is the peptidoform table or an imported library's precursors. Returns the survivors
/// table, or `None` when neither is set.
pub fn run_before_prediction(
    cfg: &Config,
    config_hash: &str,
    ms2: &str,
    table: &str,
    out_dir: &str,
) -> Result<Option<String>> {
    let p = &cfg.prescreen;
    if !p.tag_prefilter && !p.score_before_prediction {
        return Ok(None);
    }
    let mut parts: Vec<String> = Vec::new();
    if p.tag_prefilter {
        let out = format!("{out_dir}/tag_prefilter_survivors.parquet");
        info!(stage = %"tag-prefilter", "run: stage start");
        run(PrescreenParams {
            ms2,
            library_precursors: table,
            run_windows: None,
            ms1: None,
            out: &out,
            config: cfg,
            config_hash,
            sample_candidates: 0,
            tags_only: true,
        })?;
        parts.push(out);
    }
    if p.score_before_prediction {
        let out = format!("{out_dir}/score_prefilter_survivors.parquet");
        info!(stage = %"score-prefilter", "run: stage start");
        run(PrescreenParams {
            ms2,
            library_precursors: table,
            run_windows: None,
            ms1: None,
            out: &out,
            config: cfg,
            config_hash,
            sample_candidates: 0,
            tags_only: false,
        })?;
        parts.push(out);
    }
    if parts.len() == 1 {
        return Ok(parts.pop());
    }
    // Both: a candidate must pass both.
    let a = TableFile::open(&parts[0])?;
    let a_ids = a.u32("candidate_id")?;
    let b_ids: std::collections::HashSet<u32> = TableFile::open(&parts[1])?
        .u32("candidate_id")?
        .into_iter()
        .collect();
    let (lab, dict) = a.str_interned("label")?;
    let mut keep: Vec<(u32, String)> = a_ids
        .iter()
        .zip(&lab)
        .filter(|(c, _)| b_ids.contains(c))
        .map(|(&c, &l)| (c, dict[l as usize].clone()))
        .collect();
    keep.sort_unstable();
    let out = format!("{out_dir}/prediction_prefilter_survivors.parquet");
    write_table(
        &out,
        vec![
            Col::U32("candidate_id".into(), keep.iter().map(|k| k.0).collect()),
            Col::Str("label".into(), keep.into_iter().map(|k| k.1).collect()),
        ],
    )?;
    info!(out = %out, "prescreen: tag prefilter and score prefilter intersected");
    Ok(Some(out))
}

/// Copy the rows of `input` whose `id_col` is among the survivors' `candidate_id`s to `out`,
/// streaming, in input order and with the input schema. Returns the rows written.
pub fn filter_rows(input: &str, id_col: &str, survivors: &str, out: &str) -> Result<u64> {
    use arrow::array::{Array, BooleanArray, UInt32Array};
    use arrow::compute::filter_record_batch;
    use mumdia_io::table::BatchWriter;
    mumdia_io::refuse_output_over_input(out, &[("input", input), ("survivors", survivors)])?;
    let ids = TableFile::open(survivors)?.u32("candidate_id")?;
    let max = ids.iter().copied().max().unwrap_or(0) as usize;
    let mut keep = vec![false; max + 1];
    for &i in &ids {
        keep[i as usize] = true;
    }
    drop(ids);
    let t = TableFile::open(input)?;
    let reader = t.batches(None, 1 << 16)?;
    let schema = reader.schema();
    let ix = schema
        .index_of(id_col)
        .map_err(|_| anyhow::anyhow!("{input}: no column '{id_col}'"))?;
    let mut w = BatchWriter::with_row_group_rows(out, schema.clone(), 1 << 17)?;
    let mut rows = 0u64;
    for b in reader {
        let b = b?;
        let col = b
            .column(ix)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| anyhow::anyhow!("{input}: '{id_col}' is not u32"))?;
        let mask: BooleanArray = (0..col.len())
            .map(|i| Some(keep.get(col.value(i) as usize).copied().unwrap_or(false)))
            .collect();
        let kept = filter_record_batch(&b, &mask)?;
        if kept.num_rows() > 0 {
            rows += kept.num_rows() as u64;
            w.write(&kept)?;
        }
    }
    w.close()?;
    Ok(rows)
}

pub fn run(p: PrescreenParams) -> Result<PrescreenSummary> {
    let t0 = Instant::now();
    let mut inputs = vec![("--lib-precursors", p.library_precursors), ("--ms2", p.ms2)];
    if let Some(rw) = p.run_windows {
        inputs.push(("--run-windows", rw));
    }
    mumdia_io::refuse_output_over_input(p.out, &inputs)?;
    // The tag prefilter is tag retrieval without the rescue route and without a score.
    let forced;
    let config = if p.tags_only {
        let mut c = p.config.clone();
        c.prescreen.retrieval = PrescreenRetrieval::Tags;
        c.prescreen.rescue = false;
        c.prescreen.complement_bonus = 0.0;
        c.prescreen.tag_bonus = 0.0;
        c.prescreen.fasta_bonus = 0.0;
        c.prescreen.flank_bonus = 0.0;
        c.prescreen.trace.enabled = false;
        c.prescreen.mass_hypotheses.enabled = false;
        c.prescreen.localization = PrescreenLocalization::Own;
        forced = c;
        &forced
    } else {
        p.config
    };
    let cfg = &config.prescreen;
    let target = cfg.effective_target();
    let scope_mods = resolve_scope(&cfg.scope_mods)?;

    let sp = Spectra::load(p.ms2, cfg.top_peaks, cfg.trace.enabled)?;
    info!(
        spectra = sp.rt.len(),
        peaks = sp.mz.len(),
        windows = sp.groups.len(),
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "prescreen: loaded spectra"
    );

    let lib = TableFile::open(p.library_precursors)?;
    // A library carries `candidate_id` and `precursor_mz`. The peptidoform table written
    // before prediction carries neither: its row key is `id` and the precursor m/z follows
    // from the sequence and charge, so the screen can run before any predictor does.
    let id_col = if lib.has_column("candidate_id") {
        "candidate_id"
    } else {
        "id"
    };
    let mut cid = lib.u32(id_col)?;
    let (pform_off, pform_data) = lib.str_flat("peptidoform")?;
    let mut charge = lib.i32("charge")?;
    let mut pmz = if lib.has_column("precursor_mz") {
        lib.f64("precursor_mz")?
    } else {
        (0..lib.nrows)
            .into_par_iter()
            .map(|i| {
                let s = &pform_data[pform_off[i]..pform_off[i + 1]];
                parse_peptidoform(s.strip_prefix("DECOY_").unwrap_or(s))
                    .map(|q| q.precursor_mz(charge[i]))
                    .unwrap_or(f64::NAN)
            })
            .collect()
    };
    let (mut label_id, label_dict) = lib.str_interned("label")?;
    let mut rows: Vec<usize> = (0..lib.nrows).collect();
    drop(lib);
    if p.sample_candidates > 0 && p.sample_candidates < rows.len() {
        // Keep a candidate when a seeded hash of its id falls below the sampling fraction:
        // independent of label and row order.
        let frac = p.sample_candidates as f64 / rows.len() as f64;
        let salt = cfg.seed ^ 0x5A4D_504C_455F_5345;
        rows.retain(|&i| {
            let mut z = salt ^ (cid[i] as u64);
            z = (z ^ (z >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            z = (z ^ (z >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
            z ^= z >> 33;
            ((z >> 11) as f64 / (1u64 << 53) as f64) < frac
        });
        cid = rows.iter().map(|&i| cid[i]).collect();
        pmz = rows.iter().map(|&i| pmz[i]).collect();
        charge = rows.iter().map(|&i| charge[i]).collect();
        label_id = rows.iter().map(|&i| label_id[i]).collect();
        warn!(
            sampled = rows.len(),
            "prescreen: evaluation sample; the survivors cover the sample only"
        );
    }
    let pform = |i: usize| {
        let r = rows[i];
        let s = &pform_data[pform_off[r]..pform_off[r + 1]];
        s.strip_prefix("DECOY_").unwrap_or(s)
    };
    let n = rows.len();

    // Per-candidate RT bounds, joined by candidate_id through a dense lookup.
    let maxc = cid.iter().copied().max().unwrap_or(0) as usize;
    let mut lo_by = vec![f64::NAN; maxc + 1];
    let mut hi_by = vec![f64::NAN; maxc + 1];
    if let Some(path) = p.run_windows {
        let rw = TableFile::open(path)?;
        let r_cid = rw.u32("candidate_id")?;
        let r_lo = rw.f64("rt_lo")?;
        let r_hi = rw.f64("rt_hi")?;
        for i in 0..rw.nrows {
            let c = r_cid[i] as usize;
            if c <= maxc {
                lo_by[c] = r_lo[i];
                hi_by[c] = r_hi[i];
            }
        }
    }
    let slack = cfg.rt_slack_s;
    let cands: Vec<Option<Candidate>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if residue_masses(pform(i))?.len() < 2 {
                return None;
            }
            let c = cid[i] as usize;
            let (lo, hi) = (lo_by[c], hi_by[c]);
            // An unbounded or missing window is "search the whole gradient" (docs/31 F1).
            let rt = (lo.is_finite() && hi.is_finite()).then_some((lo - slack, hi + slack));
            Some(Candidate {
                nz: cfg.max_frag_charge.min(charge[i]).max(1),
                rt,
            })
        })
        .collect();
    drop(lo_by);
    drop(hi_by);
    let unparsed = cands.iter().filter(|c| c.is_none()).count();
    let unbounded = cands
        .iter()
        .filter(|c| matches!(c, Some(Candidate { rt: None, .. })))
        .count();
    if unbounded > 0 {
        warn!(
            candidates = unbounded,
            of = n,
            "prescreen: these candidates have no bounded RT window and were scored over the \
             whole gradient"
        );
    }
    if unparsed > 0 {
        warn!(
            candidates = unparsed,
            "prescreen: peptidoforms the mass model cannot parse pass through unscored"
        );
    }

    let t1 = Instant::now();
    let members = group_members(&sp, &cands, &pmz);
    let masses_of = |i: usize| residue_masses(pform(i)).unwrap_or_default();
    let (base_score, repeat_score) = if p.tags_only {
        (
            cands
                .iter()
                .map(|c| if c.is_some() { 0.0 } else { f64::NAN })
                .collect(),
            Vec::new(),
        )
    } else {
        score_all(&sp, &cands, &members, cfg, &masses_of)
    };
    let scoring_ms = t1.elapsed().as_millis() as u64;
    info!(
        candidates = n,
        elapsed_ms = scoring_ms,
        "prescreen: scored candidates"
    );

    // ---- optional components (nothing runs under the defaults) ----
    let t2 = Instant::now();
    let ext_out = if ext::needs_tags(config) {
        let pforms: Vec<&str> = (0..n).map(pform).collect();
        let alpha = tags::Alphabet::from_config(config)?;
        let view = ext::tag_view(&pforms, &charge, &alpha, config);
        let ms1 = match (
            cfg.mass_hypotheses.enabled && cfg.mass_hypotheses.ms1,
            p.ms1,
        ) {
            (true, Some(path)) => Some(load_ms1(path)?),
            (true, None) => {
                anyhow::bail!("prescreen.mass_hypotheses.ms1 needs the run's MS1 spectra (--ms1)")
            }
            _ => None,
        };
        Some(ext::run(
            &sp,
            &cands,
            &members,
            &view,
            &alpha,
            config,
            ms1.as_deref(),
            &masses_of,
        ))
    } else {
        None
    };
    let ext_ms = t2.elapsed().as_millis() as u64;
    drop(cands);
    drop(sp);

    // Scope: unparsed candidates are out of scope and pass through.
    let scoped: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| {
            !base_score[i].is_nan()
                && match cfg.scope {
                    PrescreenScope::All => true,
                    PrescreenScope::Modified => {
                        in_scope(pform(i), &scope_mods, cfg.scope_match).unwrap_or(false)
                    }
                }
        })
        .collect();
    let calib: Vec<bool> = (0..n)
        .map(|i| scoped[i] && is_calibration(cfg.seed, cid[i]))
        .collect();
    let n_cal = calib.iter().filter(|&&c| c).count();
    let enough = n_cal >= cfg.min_calibration.max(1);

    // Combined score: `base / q95(base) + sum_k w_k c_k / q95(c_k)` over the calibration half,
    // the prototype's scaling, when any component weight is positive.
    let weights = [
        cfg.tag_bonus,
        cfg.fasta_bonus,
        cfg.complement_bonus,
        cfg.flank_bonus,
        cfg.trace.bonus,
        cfg.mass_hypotheses.bonus,
    ];
    // Repeat bonus (phase 11): `D / q95(D) + w R / q95(R)` over the calibration half, before any
    // optional component, so the score below is the base every later combination scales.
    let mut repeat_scales: Option<(f64, f64)> = None;
    let base_score: Vec<f64> = if cfg.repeat_bonus > 0.0 && enough && !p.tags_only {
        let q95 = |v: &[f64]| {
            let mut c: Vec<f64> = (0..n).filter(|&i| calib[i]).map(|i| v[i]).collect();
            c.sort_by(f64::total_cmp);
            quantile_linear(&c, 0.95).max(1e-12)
        };
        let (sd, sr) = (q95(&base_score), q95(&repeat_score));
        repeat_scales = Some((sd, sr));
        (0..n)
            .map(|i| {
                if base_score[i].is_nan() {
                    f64::NAN
                } else {
                    base_score[i] / sd + cfg.repeat_bonus * repeat_score[i] / sr
                }
            })
            .collect()
    } else {
        base_score
    };
    let mut score = base_score.clone();
    let mut scales: Option<(f64, Vec<f64>)> = None;
    if let Some(e) = &ext_out {
        if enough && weights.iter().any(|&w| w > 0.0) {
            let q95 = |f: &dyn Fn(usize) -> f64| {
                let mut v: Vec<f64> = (0..n).filter(|&i| calib[i]).map(f).collect();
                v.sort_by(f64::total_cmp);
                quantile_linear(&v, 0.95).max(1e-12)
            };
            let bs = q95(&|i| base_score[i]);
            let cs: Vec<f64> = (0..ext::N_EXT)
                .map(|k| q95(&|i| e.components[i][k]))
                .collect();
            for i in 0..n {
                if score[i].is_nan() {
                    continue;
                }
                let mut s = base_score[i] / bs;
                for k in 0..ext::N_EXT {
                    if weights[k] > 0.0 {
                        s += weights[k] * e.components[i][k] / cs[k];
                    }
                }
                score[i] = s;
            }
            scales = Some((bs, cs));
        }
    }

    // Retrieval: without the rescue route an unretrieved candidate is dropped; its score
    // before that is kept to count retrieval losses separately from scoring losses.
    let retrieved: Option<Vec<bool>> = ext_out.as_ref().and_then(|e| e.retrieved.clone());
    let pre_retrieval = score.clone();
    if let Some(r) = &retrieved {
        if !cfg.rescue {
            for i in 0..n {
                if !r[i] && !score[i].is_nan() {
                    score[i] = f64::NEG_INFINITY;
                }
            }
        }
    }

    // Localization policy over modification siblings.
    let mut sibling_stats = (0usize, 0usize);
    if cfg.localization != PrescreenLocalization::Own {
        let keys: Vec<Option<String>> = (0..n)
            .into_par_iter()
            .map(|i| sibling_key(pform(i), charge[i], label_id[i]))
            .collect();
        let mut best: HashMap<&str, (f64, usize)> = HashMap::new();
        for i in 0..n {
            if let (Some(k), false) = (&keys[i], score[i].is_nan()) {
                let e = best.entry(k.as_str()).or_insert((f64::NEG_INFINITY, 0));
                e.0 = e.0.max(score[i]);
                e.1 += 1;
            }
        }
        sibling_stats.0 = best.values().filter(|v| v.1 > 1).count();
        for i in 0..n {
            let (Some(k), false) = (&keys[i], score[i].is_nan()) else {
                continue;
            };
            let (top, size) = best[k.as_str()];
            if size > 1 && score[i] == top {
                sibling_stats.1 += 1;
            }
            match cfg.localization {
                PrescreenLocalization::FamilySupport => score[i] = top,
                PrescreenLocalization::BestSite => {
                    if score[i] < top {
                        score[i] = f64::NEG_INFINITY;
                    }
                }
                PrescreenLocalization::Own => {}
            }
        }
    }

    let mut cal_scores: Vec<f64> = (0..n).filter(|&i| calib[i]).map(|i| score[i]).collect();
    cal_scores.sort_by(f64::total_cmp);
    let (cutoff, bypass) = if p.tags_only {
        (None, None)
    } else if !enough {
        (
            None,
            Some(format!(
                "{} in-scope calibration candidates, fewer than prescreen.min_calibration {}; \
                 every candidate kept",
                cal_scores.len(),
                cfg.min_calibration
            )),
        )
    } else {
        (Some(quantile_higher(&cal_scores, target)), None)
    };
    if let Some(r) = &bypass {
        warn!("prescreen: bypassed: {r}");
    }
    let keep: Vec<bool> = if p.tags_only {
        // Retrieved, or not judgeable by tags (an unparsed peptidoform, or no expressible
        // trimer, which retrieval already marks retrieved).
        let r = ext_out
            .as_ref()
            .and_then(|e| e.retrieved.as_ref())
            .expect("tags_only forces tag retrieval");
        (0..n).map(|i| !scoped[i] || r[i]).collect()
    } else {
        (0..n)
            .map(|i| !scoped[i] || cutoff.is_none_or(|c| score[i] > c))
            .collect()
    };

    let label = |i: usize| label_dict[label_id[i] as usize].as_str();
    let mut surv: Vec<(u32, &str)> = (0..n)
        .filter(|&i| keep[i])
        .map(|i| (cid[i], label(i)))
        .collect();
    surv.sort_unstable();
    let n_scoped = scoped.iter().filter(|&&s| s).count();
    let n_cal = cal_scores.len();
    let rep_n = (0..n).filter(|&i| scoped[i] && !calib[i]).count();
    let rep_dropped = (0..n)
        .filter(|&i| scoped[i] && !calib[i] && !keep[i])
        .count();
    let cal_dropped = (0..n).filter(|&i| calib[i] && !keep[i]).count();
    let dropped = n - surv.len();
    let frac = |a: usize, b: usize| if b > 0 { a as f64 / b as f64 } else { f64::NAN };
    let by_label = |want: bool| -> (usize, usize) {
        let mut tot = 0;
        let mut kept = 0;
        for i in 0..n {
            if scoped[i] && (label(i) == "target") == want {
                tot += 1;
                kept += keep[i] as usize;
            }
        }
        (tot, kept)
    };
    let (t_tot, t_kept) = by_label(true);
    let (d_tot, d_kept) = by_label(false);
    let n_t = surv.iter().filter(|(_, l)| *l == "target").count();
    let n_d = surv.len() - n_t;
    info!(
        screened = n,
        in_scope = n_scoped,
        survivors = surv.len(),
        cutoff = cutoff.unwrap_or(f64::NAN),
        scope_reduction = frac(dropped, n_scoped),
        reporting_reduction = frac(rep_dropped, rep_n),
        target_retention = frac(t_kept, t_tot),
        decoy_retention = frac(d_kept, d_tot),
        "prescreen: filtered candidates"
    );
    if n > 0 && surv.is_empty() {
        anyhow::bail!("prescreen screened {n} candidates and none survived");
    }
    if surv.len() > 1000 && (n_t == 0 || n_d == 0) {
        anyhow::bail!(
            "prescreen survivors are single-label (targets {n_t}, decoys {n_d}); \
             target-decoy exchangeability is destroyed and downstream FDR would be invalid"
        );
    }

    let rows = write_table(
        p.out,
        vec![
            Col::U32(
                "candidate_id".into(),
                surv.iter().map(|(c, _)| *c).collect(),
            ),
            Col::Str(
                "label".into(),
                surv.iter().map(|(_, l)| (*l).to_string()).collect(),
            ),
        ],
    )?;
    if cfg.write_scores {
        let path = format!("{}.scores.parquet", p.out.trim_end_matches(".parquet"));
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by_key(|&i| cid[i]);
        let srows = write_table(
            &path,
            vec![
                Col::U32("candidate_id".into(), idx.iter().map(|&i| cid[i]).collect()),
                Col::Str(
                    "label".into(),
                    idx.iter().map(|&i| label(i).to_string()).collect(),
                ),
                Col::F64("score".into(), idx.iter().map(|&i| score[i]).collect()),
                Col::Bool("in_scope".into(), idx.iter().map(|&i| scoped[i]).collect()),
                Col::Bool(
                    "calibration".into(),
                    idx.iter().map(|&i| calib[i]).collect(),
                ),
                Col::Bool("kept".into(), idx.iter().map(|&i| keep[i]).collect()),
                Col::F64(
                    "base_score".into(),
                    idx.iter().map(|&i| base_score[i]).collect(),
                ),
            ]
            .into_iter()
            .chain(
                retrieved
                    .iter()
                    .map(|r| Col::Bool("retrieved".into(), idx.iter().map(|&i| r[i]).collect())),
            )
            .chain(ext_out.iter().flat_map(|e| {
                (0..ext::N_EXT).map(|k| {
                    Col::F64(
                        format!("component_{}", ext::NAMES[k]),
                        idx.iter().map(|&i| e.components[i][k]).collect(),
                    )
                })
            }))
            .collect(),
        )?;
        ArtifactReport {
            logical_name: artifact::PRESCREEN_SCORES.0.to_string(),
            schema_name: artifact::PRESCREEN_SCORES.0.to_string(),
            schema_version: artifact::PRESCREEN_SCORES.1,
            stage: "prescreen".to_string(),
            rows: srows,
            content_hash: mumdia_io::hash::blake3_file(&path)?,
            params: json!({ "survivors": p.out, "config_hash": p.config_hash }),
            stats: Default::default(),
            model_identity: None,
            elapsed_ms: t0.elapsed().as_millis(),
        }
        .write_for(&path)?;
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats: BTreeMap<String, serde_json::Value> = Default::default();
    stats.insert("screened".into(), json!(n));
    stats.insert("in_scope".into(), json!(n_scoped));
    stats.insert("survivors".into(), json!(surv.len()));
    stats.insert("targets".into(), json!(n_t));
    stats.insert("decoys".into(), json!(n_d));
    stats.insert("target_decoy_ratio".into(), json!(frac(n_t, n_d)));
    stats.insert("unparsed_passed_through".into(), json!(unparsed));
    stats.insert("rt_unbounded".into(), json!(unbounded));
    stats.insert("calibration_n".into(), json!(n_cal));
    stats.insert("reporting_n".into(), json!(rep_n));
    stats.insert("target".into(), json!(target));
    stats.insert("cutoff".into(), json!(cutoff));
    if cutoff == Some(f64::NEG_INFINITY) {
        stats.insert(
            "cutoff_note".into(),
            json!("-inf: retrieval alone removed more than the target share"),
        );
    }
    stats.insert("bypass_reason".into(), json!(bypass));
    stats.insert(
        "calibration_reduction".into(),
        json!(frac(cal_dropped, n_cal)),
    );
    stats.insert(
        "reporting_reduction".into(),
        json!(frac(rep_dropped, rep_n)),
    );
    stats.insert("scope_reduction".into(), json!(frac(dropped, n_scoped)));
    stats.insert("total_reduction".into(), json!(frac(dropped, n)));
    stats.insert("target_retention".into(), json!(frac(t_kept, t_tot)));
    stats.insert("decoy_retention".into(), json!(frac(d_kept, d_tot)));
    stats.insert("scoring_ms".into(), json!(scoring_ms));
    stats.insert("extension_ms".into(), json!(ext_ms));
    stats.insert("sample_candidates".into(), json!(p.sample_candidates));
    stats.insert("tags_only".into(), json!(p.tags_only));
    if let Some((sd, sr)) = repeat_scales {
        stats.insert("repeat_scales_q95".into(), json!([sd, sr]));
    }
    if let Some((bs, cs)) = &scales {
        stats.insert("base_scale_q95".into(), json!(bs));
        stats.insert("component_scales_q95".into(), json!(cs));
    }
    if let Some(r) = &retrieved {
        // Losses are judged against the cutoff the score alone would set: without the rescue
        // route an unretrieved candidate scores -inf, and when retrieval removes more than the
        // target share the calibrated cutoff itself becomes -inf.
        let cut = if enough {
            let mut v: Vec<f64> = (0..n)
                .filter(|&i| calib[i])
                .map(|i| pre_retrieval[i])
                .collect();
            v.sort_by(f64::total_cmp);
            quantile_higher(&v, target)
        } else {
            f64::NEG_INFINITY
        };
        stats.insert("score_only_cutoff".into(), json!(cut));
        let unret: Vec<usize> = (0..n).filter(|&i| scoped[i] && !r[i]).collect();
        stats.insert("retrieved".into(), json!((0..n).filter(|&i| r[i]).count()));
        stats.insert("not_retrieved".into(), json!(unret.len()));
        // Candidates the score would keep but retrieval missed: rescued when the rescue route
        // is on, lost otherwise. Kept apart from candidates the score itself rejected.
        let would_keep = unret.iter().filter(|&&i| pre_retrieval[i] > cut).count();
        if cfg.rescue {
            stats.insert("rescued_by_score".into(), json!(would_keep));
            stats.insert("retrieval_losses".into(), json!(0));
        } else {
            stats.insert("rescued_by_score".into(), json!(0));
            stats.insert("retrieval_losses".into(), json!(would_keep));
        }
        stats.insert(
            "scoring_losses".into(),
            json!((0..n).filter(|&i| scoped[i] && r[i] && !keep[i]).count()),
        );
    }
    if cfg.localization != PrescreenLocalization::Own {
        stats.insert("sibling_families".into(), json!(sibling_stats.0));
        stats.insert(
            "sibling_forms_at_family_best".into(),
            json!(sibling_stats.1),
        );
    }
    if let Some(e) = &ext_out {
        for (k, v) in &e.stats {
            stats.insert(k.clone(), v.clone());
        }
        if cfg.mass_hypotheses.enabled {
            let path = format!(
                "{}.mass_hypotheses.parquet",
                p.out.trim_end_matches(".parquet")
            );
            let h = &e.hypotheses;
            write_table(
                &path,
                vec![
                    Col::U32("spectrum".into(), h.iter().map(|x| x.0).collect()),
                    Col::F64("rt_seconds".into(), h.iter().map(|x| x.1).collect()),
                    Col::F64(
                        "neutral_mass".into(),
                        h.iter().map(|x| x.2.neutral).collect(),
                    ),
                    Col::I32(
                        "complements".into(),
                        h.iter().map(|x| x.2.complements as i32).collect(),
                    ),
                    Col::F64("fit".into(), h.iter().map(|x| x.2.quality).collect()),
                    Col::U32("tag_key".into(), h.iter().map(|x| x.2.key).collect()),
                    Col::I32(
                        "precursor_charge".into(),
                        h.iter().map(|x| x.2.precursor_charge).collect(),
                    ),
                    Col::I32("ms1_link".into(), h.iter().map(|x| x.3 as i32).collect()),
                ],
            )?;
        }
    }
    ArtifactReport {
        logical_name: "prescreen_survivors".to_string(),
        schema_name: artifact::PRESCAN_SURVIVORS.0.to_string(),
        schema_version: artifact::PRESCAN_SURVIVORS.1,
        stage: "prescreen".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "ms2": p.ms2,
            "library_precursors": p.library_precursors,
            "run_windows": p.run_windows,
            "prescreen": cfg,
            "effective_target": if p.tags_only { None } else { Some(target) },
            "config_hash": p.config_hash,
        }),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;
    info!(
        out = p.out,
        rows,
        elapsed_ms = elapsed as u64,
        "prescreen: done"
    );
    Ok(PrescreenSummary {
        screened: n as u64,
        kept: rows,
        cutoff,
        bypass_reason: bypass,
    })
}

/// numpy's default (linear) quantile on a sorted slice.
pub(crate) fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = (sorted.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Modification siblings: same residues, charge, label and modification composition (the
/// sorted deltas, terminal ones included), so they differ only in localization.
fn sibling_key(pf: &str, charge: i32, label: u32) -> Option<String> {
    let p = parse_peptidoform(pf).ok()?;
    let mut d: Vec<i64> = p
        .mods
        .iter()
        .chain([&p.n_term_mod, &p.c_term_mod])
        .filter(|&&x| x != 0.0)
        .map(|&x| (x * 1e4).round() as i64)
        .collect();
    d.sort_unstable();
    Some(format!(
        "{}/{charge}/{label}/{d:?}",
        String::from_utf8_lossy(&p.residues)
    ))
}

/// MS1 scans as `(rt, mz ascending, intensity)`, ascending RT.
fn load_ms1(path: &str) -> Result<Vec<masses::Ms1Scan>> {
    let t = TableFile::open(path)?;
    let rt = t.f64("rt_seconds")?;
    let mz = t.list_f32("mz")?;
    let it = t.list_f32("intensity")?;
    let mut v: Vec<masses::Ms1Scan> = (0..rt.len())
        .map(|s| {
            let n = mz[s].len().min(it[s].len());
            let mut p: Vec<(f64, f32)> = (0..n).map(|k| (mz[s][k] as f64, it[s][k])).collect();
            p.sort_by(|a, b| a.0.total_cmp(&b.0));
            (
                rt[s],
                p.iter().map(|x| x.0).collect(),
                p.iter().map(|x| x.1).collect(),
            )
        })
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> String {
        let dir =
            std::env::temp_dir().join(format!("mumdia_prescreen_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_str().unwrap().to_string()
    }

    fn cfg() -> PrescreenConfig {
        PrescreenConfig::default()
    }

    fn conf(c: PrescreenConfig) -> Config {
        Config {
            prescreen: c,
            ..Config::default()
        }
    }

    /// Fragment m/z of one orientation, as the scorer builds them.
    fn frags(pf: &str, nz: i32, reversed: bool) -> Vec<f64> {
        let m = residue_masses(pf).unwrap();
        let mut out = Vec::new();
        fragment_mz(&m, reversed, nz, &mut out);
        out
    }

    /// A one-window `Spectra` from `(rt, [(mz, intensity)])`, peaks sorted, nothing capped.
    fn one_window(spectra: Vec<(f64, Vec<(f32, f32)>)>) -> Spectra {
        let mut s = Spectra {
            rt: Vec::new(),
            offsets: vec![0],
            mz: Vec::new(),
            positive: Vec::new(),
            intensity: Vec::new(),
            groups: Vec::new(),
            nbin: 0,
        };
        let mut max_mz = 0.0f64;
        for (rt, mut p) in spectra {
            p.sort_by(|a, b| a.0.total_cmp(&b.0));
            s.rt.push(rt);
            for (m, i) in p {
                s.mz.push(m);
                s.positive.push(i > 0.0);
                max_mz = max_mz.max(m as f64);
            }
            s.offsets.push(s.mz.len());
        }
        s.groups.push(WindowGroup {
            lower: 0.0,
            upper: 10_000.0,
            first: 0,
            count: s.rt.len(),
        });
        s.nbin = (max_mz * 100.0).ceil() as usize + 2;
        s
    }

    /// Score one candidate against a one-window `Spectra`, the histogram built from it.
    fn score_one(
        sp: &Spectra,
        pf: &str,
        z: i32,
        rt: Option<(f64, f64)>,
        c: &PrescreenConfig,
    ) -> f64 {
        let g = &sp.groups[0];
        let hist = Histogram::build((0..sp.rt.len()).map(|s| sp.peaks(s)), sp.nbin);
        let neglnp: Vec<f64> = sp
            .mz
            .iter()
            .map(|&m| -hist.probability(m as f64).ln())
            .collect();
        let cand = Candidate {
            nz: c.max_frag_charge.min(z).max(1),
            rt,
        };
        let m = residue_masses(pf).unwrap();
        score_in_group(
            &cand,
            &m,
            sp,
            g,
            None,
            &neglnp,
            c,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .0
    }

    fn planted(pf: &str, nz: i32) -> Vec<(f32, f32)> {
        frags(pf, nz, false)
            .into_iter()
            .map(|m| (m as f32, 100.0))
            .collect()
    }

    /// Background spectra so that `p` is well below 1 for the planted peaks.
    fn with_background(mut s: Vec<(f64, Vec<(f32, f32)>)>) -> Spectra {
        for k in 0..20 {
            s.push((1000.0 + k as f64, vec![(1999.5 + k as f32, 1.0)]));
        }
        one_window(s)
    }

    #[test]
    fn the_bin_rounds_half_to_even_like_numpy_rint() {
        assert_eq!(bin_of(0.125), 12, "12.5 -> 12");
        assert_eq!(
            bin_of(0.135),
            14,
            "13.5 -> 14 (0.135 * 100 is 13.5000...02)"
        );
        assert_eq!(
            bin_of(1.005 * 1.0),
            (1.005f64 * 100.0).round_ties_even() as i64
        );
        assert_eq!(bin_of(2.5 / 100.0), 2);
        assert_eq!(bin_of(3.5 / 100.0), 4);
    }

    #[test]
    fn probability_sums_three_bins_with_the_prototype_smoothing() {
        let a: Vec<f32> = vec![500.00];
        let b: Vec<f32> = vec![500.01];
        let c: Vec<f32> = vec![500.02, 500.021];
        let h = Histogram::build([&a[..], &b[..], &c[..]].into_iter(), 60_000);
        assert_eq!(h.n_spectra, 3);
        assert_eq!(h.counts[50_002], 1, "a bin counts once per spectrum");
        // bins 50000..=50002 hold three spectra: (3 + 0.5) / (3 + 1), capped at 0.99999.
        assert!((h.probability(500.01) - 0.875).abs() < 1e-12);
        // Nothing near: (0 + 0.5) / 4.
        assert!((h.probability(700.0) - 0.125).abs() < 1e-12);
        let one: Vec<f32> = vec![500.0];
        let full = Histogram::build(std::iter::repeat_n(&one[..], 1_000_000), 60_000);
        assert_eq!(full.probability(500.0), 0.99999);
    }

    /// The prototype's `nearest_index`: closest within the tolerance, the first on a tie, and
    /// both edges inclusive.
    #[test]
    fn nearest_index_takes_the_closest_and_the_first_on_a_tie() {
        let mz: Vec<f32> = vec![99.0, 100.0, 100.25, 100.75, 101.0];
        assert_eq!(nearest_index(&mz, 100.5, 0.25), Some(2), "tie -> first");
        assert_eq!(nearest_index(&mz, 100.1, 0.25), Some(1));
        assert_eq!(nearest_index(&mz, 100.0, 0.0), Some(1), "inclusive edge");
        assert_eq!(nearest_index(&mz, 102.0, 0.5), None);
    }

    /// m/z is stored as f32, and every comparison is in f64 on the widened stored value. The
    /// f32 nearest to 500.005 is 500.00500488..., so it lies outside 500.0 + 0.005 in f64 and
    /// must not match, although the same test done in f32 arithmetic would accept it.
    #[test]
    fn f32_peaks_are_compared_in_f64_at_the_tolerance_boundary() {
        let edge = 500.005f32;
        assert!(edge as f64 > 500.005);
        assert!(
            edge <= 500.0f32 + 0.005f32,
            "f32 arithmetic would accept it"
        );
        assert_eq!(nearest_index(&[edge], 500.0, 0.005), None);
        let inside = 500.004f32;
        assert_eq!(nearest_index(&[inside], 500.0, 0.005), Some(0));

        // Randomised parity with an f64 reference on the widened values.
        let mut s = 0x1234_5678u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..2000 {
            let mut mz: Vec<f32> = (0..50).map(|_| (300.0 + next() * 2.0) as f32).collect();
            mz.sort_by(f32::total_cmp);
            let wide: Vec<f64> = mz.iter().map(|&m| m as f64).collect();
            let x = 300.0 + next() * 2.0;
            let reference = {
                let mut best = None;
                let mut err = 0.005 + 1.0;
                for (j, &m) in wide.iter().enumerate() {
                    if m >= x - 0.005 && m <= x + 0.005 && (m - x).abs() < err {
                        best = Some(j);
                        err = (m - x).abs();
                    }
                }
                best
            };
            assert_eq!(nearest_index(&mz, x, 0.005), reference);
            for &m in &mz {
                assert_eq!(
                    bin_of(m as f64),
                    (m as f64 * 100.0).round_ties_even() as i64
                );
            }
        }
    }

    /// b1 and y1 are included (unlike `ParsedPeptidoform::fragments`), b at `prefix/z + H+`
    /// and y at `(suffix + H2O)/z + H+`.
    #[test]
    fn fragments_include_b1_and_y1_at_every_allowed_charge() {
        let f = frags("PEPTIDEK", 2, false);
        assert_eq!(f.len(), 7 * 2 * 2);
        let p = residue_mass(b'P').unwrap();
        let k = residue_mass(b'K').unwrap();
        // The prototype's order: b z1 (7), y z1 (7), b z2 (7), y z2 (7).
        assert!((f[0] - (p + PROTON)).abs() < 1e-9, "b1 z1");
        assert!((f[7] - (k + WATER + PROTON)).abs() < 1e-9, "y1 z1");
        assert!((f[14] - (p / 2.0 + PROTON)).abs() < 1e-9, "b1 z2");
    }

    #[test]
    fn fragment_charges_follow_the_precursor_charge() {
        // Only the charge-2 fragments are present.
        let z2: Vec<(f32, f32)> = frags("PEPTIDEK", 2, false)
            .into_iter()
            .skip(14)
            .map(|m| (m as f32, 50.0))
            .collect();
        let sp = with_background(vec![(100.0, z2)]);
        let c = cfg();
        assert_eq!(
            score_one(&sp, "PEPTIDEK", 1, None, &c),
            0.0,
            "z1 precursor: z1 fragments only"
        );
        let s2 = score_one(&sp, "PEPTIDEK", 2, None, &c);
        assert!(s2 > 0.0);
        assert_eq!(
            score_one(&sp, "PEPTIDEK", 3, None, &c),
            s2,
            "capped at max_frag_charge 2"
        );
        let mut c1 = cfg();
        c1.max_frag_charge = 1;
        assert_eq!(score_one(&sp, "PEPTIDEK", 3, None, &c1), 0.0);
    }

    #[test]
    fn a_peak_hit_by_several_fragments_counts_once() {
        let mz: Vec<f32> = vec![400.0, 600.0];
        let pos = vec![true, true];
        let neg = vec![2.0, 3.0];
        let c = cfg();
        let one = spectrum_score(
            &[600.001],
            &mz,
            &pos,
            &neg,
            1.0,
            &c,
            &mut Vec::new(),
            None,
            None,
        );
        let twice = spectrum_score(
            &[600.001, 599.998, 600.0],
            &mz,
            &pos,
            &neg,
            1.0,
            &c,
            &mut Vec::new(),
            None,
            None,
        );
        assert_eq!(one, 3.0);
        assert_eq!(twice, one);
        let both = spectrum_score(
            &[600.0, 400.0],
            &mz,
            &pos,
            &neg,
            2.0,
            &c,
            &mut Vec::new(),
            None,
            None,
        );
        assert_eq!(both, 3.0 / 2.0 + 2.0 / 2.0);
    }

    #[test]
    fn low_mz_peaks_carry_the_low_mz_weight() {
        let mz: Vec<f32> = vec![250.0, 600.0];
        let pos = vec![true, true];
        let neg = vec![2.0, 3.0];
        let c = cfg();
        let s = spectrum_score(
            &[250.0, 600.0],
            &mz,
            &pos,
            &neg,
            2.0,
            &c,
            &mut Vec::new(),
            None,
            None,
        );
        assert_eq!(s, 2.0 / 2.0 * 0.5 + 3.0 / 2.0);
    }

    /// The prototype matches the nearest peak and only then requires a positive intensity, so a
    /// zero-intensity nearest peak hides a positive one further away.
    #[test]
    fn a_zero_intensity_nearest_peak_is_not_a_match() {
        let mz: Vec<f32> = vec![600.000, 600.004];
        let neg = vec![1.0, 1.0];
        let c = cfg();
        let s = spectrum_score(
            &[600.001],
            &mz,
            &[false, true],
            &neg,
            1.0,
            &c,
            &mut Vec::new(),
            None,
            None,
        );
        assert_eq!(s, 0.0);
        let s = spectrum_score(
            &[600.003],
            &mz,
            &[false, true],
            &neg,
            1.0,
            &c,
            &mut Vec::new(),
            None,
            None,
        );
        assert_eq!(s, 1.0);
    }

    #[test]
    fn the_score_is_the_maximum_over_eligible_spectra_inside_the_rt_bounds() {
        let full = planted("PEPTIDEK", 2);
        let half: Vec<(f32, f32)> = full.iter().step_by(2).copied().collect();
        let sp = with_background(vec![(100.0, half), (200.0, full)]);
        let c = cfg();
        let all = score_one(&sp, "PEPTIDEK", 2, None, &c);
        let only_half = score_one(&sp, "PEPTIDEK", 2, Some((50.0, 150.0)), &c);
        assert!(all > only_half && only_half > 0.0);
        assert_eq!(
            score_one(&sp, "PEPTIDEK", 2, Some((200.0, 200.0)), &c),
            all,
            "inclusive bounds"
        );
        assert_eq!(score_one(&sp, "PEPTIDEK", 2, Some((300.0, 400.0)), &c), 0.0);
    }

    /// With both orientations the score of a residue array equals that of its reversal, so a
    /// reversed decoy scores exactly as its target: the prototype's 100-candidate check.
    #[test]
    fn both_orientations_make_the_score_reversal_invariant() {
        let sp = with_background(vec![
            (100.0, planted("PEPTIDEK", 2)),
            (110.0, planted("LGEYGFQNALIVR", 2)),
        ]);
        let c = cfg();
        for (pf, rev) in [("PEPTIDEK", "KEDITPEP"), ("LGEYGFQNALIVR", "RVILANQFGYEGL")] {
            let a = score_one(&sp, pf, 2, None, &c);
            let b = score_one(&sp, rev, 2, None, &c);
            assert!(a > 0.0);
            assert_eq!(a, b, "{pf}");
        }
        let mut one = cfg();
        one.both_orientations = false;
        assert!(
            score_one(&sp, "KEDITPEP", 2, None, &one) < score_one(&sp, "PEPTIDEK", 2, None, &one)
        );
    }

    /// Uncapped by default: a fragment peak ranked below the 300 most intense peaks still
    /// counts; `top_peaks = 300` reproduces the capped comparison and loses it.
    #[test]
    fn peaks_ranked_below_the_cap_count_only_when_uncapped() {
        let dir = scratch("cap");
        let ms2 = format!("{dir}/ms2.parquet");
        let mut mz: Vec<f32> = (0..400).map(|i| 1000.0 + i as f32 * 0.37).collect();
        let mut it: Vec<f32> = vec![1000.0; 400];
        for m in frags("PEPTIDEK", 2, false) {
            mz.push(m as f32);
            it.push(1.0);
        }
        let n = 3;
        write_table(
            &ms2,
            vec![
                Col::F64("rt_seconds".into(), vec![10.0, 20.0, 30.0]),
                Col::F64("window_lower".into(), vec![400.0; n]),
                Col::F64("window_upper".into(), vec![500.0; n]),
                Col::ListF32("mz".into(), vec![mz.clone(), vec![2000.0], vec![2001.0]]),
                Col::ListF32("intensity".into(), vec![it.clone(), vec![1.0], vec![1.0]]),
            ],
        )
        .unwrap();
        let unc = Spectra::load(&ms2, 0, false).unwrap();
        let cap = Spectra::load(&ms2, 300, false).unwrap();
        assert_eq!(unc.mz.len(), 400 + 28 + 2);
        assert_eq!(cap.mz.len(), 300 + 2);
        let c = cfg();
        assert!(score_one(&unc, "PEPTIDEK", 2, None, &c) > 0.0);
        assert_eq!(score_one(&cap, "PEPTIDEK", 2, None, &c), 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantile_higher_matches_numpy() {
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        // np.quantile(range(1, 11), q, method="higher")
        assert_eq!(quantile_higher(&x, 0.54), 6.0);
        assert_eq!(quantile_higher(&x, 0.5), 6.0);
        assert_eq!(quantile_higher(&x, 0.0), 1.0);
        assert_eq!(quantile_higher(&x, 0.9), 10.0);
        assert_eq!(quantile_higher(&x[..4], 0.5), 3.0);
        assert_eq!(quantile_higher(&[7.0], 0.75), 7.0);
    }

    #[test]
    fn the_calibration_split_is_seeded_and_near_half() {
        let n = (0..100_000u32).filter(|&c| is_calibration(7, c)).count();
        assert!((49_000..51_000).contains(&n), "{n}");
        let a: Vec<bool> = (0..1000u32).map(|c| is_calibration(7, c)).collect();
        let b: Vec<bool> = (0..1000u32).map(|c| is_calibration(8, c)).collect();
        assert_ne!(a, b);
        assert_eq!(
            a,
            (0..1000u32)
                .map(|c| is_calibration(7, c))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn modification_scope_supports_any_and_all() {
        let ox = resolve_scope(&["M:Oxidation".into()]).unwrap();
        let both = resolve_scope(&["M:Oxidation".into(), "C:Carbamidomethyl".into()]).unwrap();
        let any = PrescreenScopeMatch::Any;
        let all = PrescreenScopeMatch::All;
        assert_eq!(in_scope("PEM[Oxidation]K", &ox, any), Some(true));
        assert_eq!(in_scope("PEMK", &ox, any), Some(false));
        assert_eq!(
            in_scope("PEM[+15.994915]K", &ox, any),
            Some(true),
            "numeric spelling"
        );
        assert_eq!(in_scope("PEM[Oxidation]K", &both, all), Some(false));
        assert_eq!(
            in_scope("PEM[Oxidation]C[Carbamidomethyl]K", &both, all),
            Some(true)
        );
        assert_eq!(in_scope("PEC[Carbamidomethyl]K", &both, any), Some(true));
        assert_eq!(
            in_scope("PEW[Oxidation]K", &ox, any),
            Some(false),
            "wrong residue"
        );
        assert!(resolve_scope(&["M:NotAModification".into()]).is_err());
    }

    /// A synthetic run: `n` random tryptic-like candidates in one isolation window, the first
    /// `planted` of them with their fragments in the spectrum at their own retention time.
    fn synthetic_run(
        dir: &str,
        n: usize,
        planted_n: usize,
        flip_labels: bool,
    ) -> (String, String, String) {
        let mut s = 0xC0FFEEu64;
        let mut next = |m: u64| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) % m
        };
        let aa = b"ACDEFGHLMNPQSTVWY";
        let mut pfs = Vec::new();
        for _ in 0..n {
            let len = 8 + next(7) as usize;
            let mut p: Vec<u8> = (0..len - 1)
                .map(|_| aa[next(aa.len() as u64) as usize])
                .collect();
            p.push(if next(2) == 0 { b'K' } else { b'R' });
            pfs.push(String::from_utf8(p).unwrap());
        }
        let mut spectra_mz = Vec::new();
        let mut spectra_int = Vec::new();
        let mut rts = Vec::new();
        for (i, pf) in pfs.iter().enumerate() {
            let mut mz: Vec<f32> = (0..200)
                .map(|_| (150.0 + next(1_350_000) as f64 / 1000.0) as f32)
                .collect();
            if i < planted_n {
                mz.extend(frags(pf, 2, false).into_iter().map(|m| m as f32));
            }
            mz.sort_by(f32::total_cmp);
            spectra_int.push(vec![10.0f32; mz.len()]);
            spectra_mz.push(mz);
            rts.push(10.0 * i as f64);
        }
        let ms2 = format!("{dir}/ms2.parquet");
        write_table(
            &ms2,
            vec![
                Col::F64("rt_seconds".into(), rts.clone()),
                Col::F64("window_lower".into(), vec![0.0; n]),
                Col::F64("window_upper".into(), vec![5000.0; n]),
                Col::ListF32("mz".into(), spectra_mz),
                Col::ListF32("intensity".into(), spectra_int),
            ],
        )
        .unwrap();
        let label = |i: usize| i.is_multiple_of(2) != flip_labels;
        let lib = format!("{dir}/lib.parquet");
        write_table(
            &lib,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::Str(
                    "peptidoform".into(),
                    pfs.iter()
                        .enumerate()
                        .map(|(i, p)| {
                            if label(i) {
                                p.clone()
                            } else {
                                format!("DECOY_{p}")
                            }
                        })
                        .collect(),
                ),
                Col::I32("charge".into(), vec![2; n]),
                Col::F64(
                    "precursor_mz".into(),
                    pfs.iter()
                        .map(|p| parse_peptidoform(p).unwrap().precursor_mz(2))
                        .collect(),
                ),
                Col::Str(
                    "label".into(),
                    (0..n)
                        .map(|i| if label(i) { "target" } else { "decoy" }.to_string())
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let rw = format!("{dir}/run_windows.parquet");
        write_table(
            &rw,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::F64("rt_lo".into(), rts.iter().map(|r| r - 1.0).collect()),
                Col::F64("rt_hi".into(), rts.iter().map(|r| r + 1.0).collect()),
            ],
        )
        .unwrap();
        (ms2, lib, rw)
    }

    fn survivors(path: &str) -> Vec<u32> {
        TableFile::open(path).unwrap().u32("candidate_id").unwrap()
    }

    #[test]
    fn end_to_end_keeps_the_planted_candidates_and_reports_the_cutoff() {
        let dir = scratch("e2e");
        let (ms2, lib, rw) = synthetic_run(&dir, 400, 120, false);
        let out = format!("{dir}/survivors.parquet");
        let mut c = cfg();
        c.write_scores = true;
        let sum = run(PrescreenParams {
            ms2: &ms2,
            library_precursors: &lib,
            run_windows: Some(&rw),
            out: &out,
            ms1: None,
            config: &conf(c.clone()),
            config_hash: "test",
            sample_candidates: 0,
            tags_only: false,
        })
        .unwrap();
        assert!(sum.bypass_reason.is_none());
        assert!(sum.cutoff.is_some());
        let kept = survivors(&out);
        assert!(
            (0..120u32).all(|i| kept.contains(&i)),
            "every planted candidate survives"
        );
        assert!(
            kept.len() < 400 / 2,
            "most unplanted candidates are removed: {}",
            kept.len()
        );
        let report: serde_json::Value =
            mumdia_io::json::read_json(&format!("{out}.report.json")).unwrap();
        let st = &report["stats"];
        assert_eq!(st["screened"], 400);
        assert!(st["calibration_n"].as_u64().unwrap() > 150);
        assert!(st["reporting_reduction"].as_f64().unwrap() > 0.5);
        let scores = TableFile::open(&format!("{dir}/survivors.scores.parquet")).unwrap();
        assert_eq!(scores.nrows, 400);

        // Label-blind: swapping every label leaves the survivor set unchanged.
        let dir2 = scratch("e2e_flip");
        let (ms2, lib, rw) = synthetic_run(&dir2, 400, 120, true);
        let out2 = format!("{dir2}/survivors.parquet");
        run(PrescreenParams {
            ms2: &ms2,
            library_precursors: &lib,
            run_windows: Some(&rw),
            out: &out2,
            ms1: None,
            config: &conf(cfg()),
            config_hash: "test",
            sample_candidates: 0,
            tags_only: false,
        })
        .unwrap();
        assert_eq!(survivors(&out2), kept);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn too_few_calibration_candidates_bypass_the_filter_with_a_reason() {
        let dir = scratch("bypass");
        let (ms2, lib, rw) = synthetic_run(&dir, 120, 10, false);
        let out = format!("{dir}/survivors.parquet");
        let sum = run(PrescreenParams {
            ms2: &ms2,
            library_precursors: &lib,
            run_windows: Some(&rw),
            out: &out,
            ms1: None,
            config: &conf(cfg()),
            config_hash: "test",
            sample_candidates: 0,
            tags_only: false,
        })
        .unwrap();
        assert!(sum.bypass_reason.unwrap().contains("min_calibration"));
        assert_eq!(survivors(&out).len(), 120);

        // Out-of-scope candidates pass through; a scope nothing matches bypasses as well.
        let mut c = cfg();
        c.scope = PrescreenScope::Modified;
        c.scope_mods = vec!["M:Oxidation".into()];
        c.min_calibration = 1;
        let sum = run(PrescreenParams {
            ms2: &ms2,
            library_precursors: &lib,
            run_windows: Some(&rw),
            out: &out,
            ms1: None,
            config: &conf(c.clone()),
            config_hash: "test",
            sample_candidates: 0,
            tags_only: false,
        })
        .unwrap();
        assert!(sum.bypass_reason.is_some());
        assert_eq!(survivors(&out).len(), 120);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn scores_table(out: &str) -> TableFile {
        TableFile::open(&format!(
            "{}.scores.parquet",
            out.trim_end_matches(".parquet")
        ))
        .unwrap()
    }

    /// Library of explicit peptidoforms, one spectrum per entry holding `planted[i]`.
    fn explicit_run(dir: &str, pfs: &[&str], planted: &[Vec<f64>]) -> (String, String, String) {
        let n = pfs.len();
        let mut s = 0xBEEFu64;
        let mut next = |m: u64| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) % m
        };
        let mut mzs = Vec::new();
        for p in planted {
            let mut mz: Vec<f32> = (0..150)
                .map(|_| (150.0 + next(1_350_000) as f64 / 1000.0) as f32)
                .collect();
            mz.extend(p.iter().map(|&m| m as f32));
            mz.sort_by(f32::total_cmp);
            mzs.push(mz);
        }
        let ints: Vec<Vec<f32>> = mzs.iter().map(|m| vec![10.0; m.len()]).collect();
        let rts: Vec<f64> = (0..n).map(|i| 10.0 * i as f64).collect();
        let ms2 = format!("{dir}/ms2.parquet");
        write_table(
            &ms2,
            vec![
                Col::F64("rt_seconds".into(), rts.clone()),
                Col::F64("window_lower".into(), vec![0.0; n]),
                Col::F64("window_upper".into(), vec![5000.0; n]),
                Col::ListF32("mz".into(), mzs),
                Col::ListF32("intensity".into(), ints),
            ],
        )
        .unwrap();
        let lib = format!("{dir}/lib.parquet");
        write_table(
            &lib,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                Col::Str(
                    "peptidoform".into(),
                    pfs.iter().map(|s| s.to_string()).collect(),
                ),
                Col::I32("charge".into(), vec![2; n]),
                Col::F64(
                    "precursor_mz".into(),
                    pfs.iter()
                        .map(|p| parse_peptidoform(p).unwrap().precursor_mz(2))
                        .collect(),
                ),
                Col::Str(
                    "label".into(),
                    (0..n)
                        .map(|i| if i % 2 == 0 { "target" } else { "decoy" }.to_string())
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let rw = format!("{dir}/rw.parquet");
        write_table(
            &rw,
            vec![
                Col::U32("candidate_id".into(), (0..n as u32).collect()),
                // Every candidate may use every spectrum, so siblings see the same evidence.
                Col::F64("rt_lo".into(), vec![-1.0; n]),
                Col::F64("rt_hi".into(), vec![1e9; n]),
            ],
        )
        .unwrap();
        (ms2, lib, rw)
    }

    /// Family support gives every localization sibling the family's best score; best-site keeps
    /// only the best and preserves ties; `own` leaves each form its own score.
    #[test]
    fn localization_policies_preserve_ties_and_keep_support_separate() {
        let dir = scratch("loc");
        // A and B: siblings, only A's full ladder is planted. C and D: siblings whose planted
        // fragments do not distinguish the sites (a tie). Labels alternate, so siblings are
        // placed on equal labels by repeating them.
        let pfs = [
            "GAM[Oxidation]STMDEK",
            "PLAINPEPTIDEK",
            "GAMSTM[Oxidation]DEK",
            "PLAINPEPTIDEKK",
            "WM[Oxidation]MWHLLLLR",
            "PLAINPEPTIDEKKK",
            "WMM[Oxidation]WHLLLLR",
        ];
        let tie_frags: Vec<f64> = frags("WM[Oxidation]MWHLLLLR", 1, false)
            .into_iter()
            .zip(frags("WMM[Oxidation]WHLLLLR", 1, false))
            .filter(|(x, y)| (x - y).abs() < 1e-9)
            .map(|(x, _)| x)
            .collect();
        assert!(!tie_frags.is_empty());
        let planted = vec![
            frags("GAM[Oxidation]STMDEK", 2, false),
            vec![],
            vec![],
            vec![],
            tie_frags,
            vec![],
            vec![],
        ];
        let (ms2, lib, rw) = explicit_run(&dir, &pfs, &planted);
        let out = format!("{dir}/s.parquet");
        let score_of = |pol: PrescreenLocalization| {
            let mut c = cfg();
            c.min_calibration = 1;
            c.write_scores = true;
            c.localization = pol;
            run(PrescreenParams {
                ms2: &ms2,
                library_precursors: &lib,
                run_windows: Some(&rw),
                ms1: None,
                out: &out,
                config: &conf(c),
                config_hash: "t",
                sample_candidates: 0,
                tags_only: false,
            })
            .unwrap();
            let t = scores_table(&out);
            (t.f64("score").unwrap(), t.f64("base_score").unwrap())
        };
        let (own, base) = score_of(PrescreenLocalization::Own);
        assert_eq!(own, base);
        assert!(base[0] > base[2] && base[2] >= 0.0);
        assert_eq!(base[4], base[6], "the tie");
        let (fam, _) = score_of(PrescreenLocalization::FamilySupport);
        assert_eq!(fam[2], base[0], "B carries its family's support");
        assert_eq!(fam[0], base[0]);
        let (best, _) = score_of(PrescreenLocalization::BestSite);
        assert_eq!(best[0], base[0]);
        assert_eq!(best[2], f64::NEG_INFINITY, "B is not the best site");
        assert_eq!(best[4], base[4]);
        assert_eq!(best[6], base[6], "ties are kept");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Tag retrieval: the planted candidates are retrieved; without the rescue route the
    /// unretrieved ones are dropped and the losses are reported apart from scoring losses; with
    /// it, the survivors equal those of `retrieval = "all"`. Every optional component runs.
    #[test]
    fn tag_retrieval_rescue_and_optional_components() {
        let dir = scratch("ret");
        let (ms2, lib, rw) = synthetic_run(&dir, 300, 80, false);
        let out = format!("{dir}/s.parquet");
        let go = |c: PrescreenConfig| {
            let sum = run(PrescreenParams {
                ms2: &ms2,
                library_precursors: &lib,
                run_windows: Some(&rw),
                ms1: None,
                out: &out,
                config: &conf(c),
                config_hash: "t",
                sample_candidates: 0,
                tags_only: false,
            })
            .unwrap();
            let rep: serde_json::Value =
                mumdia_io::json::read_json(&format!("{out}.report.json")).unwrap();
            (sum, survivors(&out), rep["stats"].clone())
        };
        let (_, all, _) = go(cfg());
        let mut c = cfg();
        c.retrieval = mumdia_core::config::PrescreenRetrieval::Tags;
        c.write_scores = true;
        let (_, rescued, st) = go(c.clone());
        assert_eq!(rescued, all, "the rescue route scores every candidate");
        let ret = scores_table(&out).u32("candidate_id").unwrap();
        assert_eq!(ret.len(), 300);
        assert!(st["retrieved"].as_u64().unwrap() >= 80);
        assert!(st["tag_paths"].as_u64().unwrap() > 0);
        c.rescue = false;
        let (_, strict, st) = go(c.clone());
        assert!(strict.iter().all(|x| rescued.contains(x)));
        assert!(
            (0..80u32).all(|i| strict.contains(&i)),
            "planted candidates are retrieved"
        );
        assert!(st.get("retrieval_losses").is_some() && st.get("scoring_losses").is_some());
        c.delayed_modforms = true;
        let (_, delayed, st) = go(c);
        assert_eq!(
            delayed, strict,
            "delayed enumeration retrieves the same set"
        );
        assert!(st["retrieval_forms_examined"].as_u64().is_some());

        let mut e = cfg();
        e.complement_bonus = 0.25;
        e.tag_bonus = 0.1;
        e.fasta_bonus = 0.1;
        e.flank_bonus = 0.1;
        e.trace.enabled = true;
        e.trace.bonus = 0.1;
        e.mass_hypotheses.enabled = true;
        e.mass_hypotheses.sample_spectra = 5;
        e.write_scores = true;
        let (sum, kept, st) = go(e);
        assert!(sum.cutoff.is_some() && !kept.is_empty());
        assert!(st["component_scales_q95"].is_array());
        let t = scores_table(&out);
        for name in ext::NAMES {
            let v = t.f64(&format!("component_{name}")).unwrap();
            assert!(v.iter().all(|x| x.is_finite() && *x >= 0.0), "{name}");
        }
        assert!(t.f64("component_tag").unwrap().iter().any(|&x| x > 0.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Before prediction: the peptidoform table (row key `id`, no `precursor_mz`) screened
    /// without RT windows keeps exactly what the library-shaped table keeps over the whole
    /// gradient.
    #[test]
    fn the_peptidoform_table_screens_without_predictions() {
        let dir = scratch("prepred");
        let (ms2, lib, _) = synthetic_run(&dir, 300, 90, false);
        let t = TableFile::open(&lib).unwrap();
        let pf = format!("{dir}/peptidoforms.parquet");
        write_table(
            &pf,
            vec![
                Col::U32("id".into(), t.u32("candidate_id").unwrap()),
                Col::Str("peptidoform".into(), {
                    let (o, d) = t.str_flat("peptidoform").unwrap();
                    (0..t.nrows)
                        .map(|i| d[o[i]..o[i + 1]].to_string())
                        .collect()
                }),
                Col::I32("charge".into(), t.i32("charge").unwrap()),
                Col::Str("label".into(), {
                    let (id, dict) = t.str_interned("label").unwrap();
                    id.iter().map(|&k| dict[k as usize].clone()).collect()
                }),
            ],
        )
        .unwrap();
        let go = |lib: &str, out: &str| {
            run(PrescreenParams {
                ms2: &ms2,
                library_precursors: lib,
                run_windows: None,
                ms1: None,
                out,
                config: &conf(cfg()),
                config_hash: "t",
                sample_candidates: 0,
                tags_only: false,
            })
            .unwrap();
            survivors(out)
        };
        let a = go(&lib, &format!("{dir}/a.parquet"));
        let b = go(&pf, &format!("{dir}/b.parquet"));
        assert_eq!(a, b);
        assert!(!a.is_empty() && a.len() < 300);
        assert!(
            (0..90u32).all(|i| a.contains(&i)),
            "planted candidates kept without RT"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The tag prefilter keeps every candidate with an observed trimer, scores nothing, and
    /// keeps the planted candidates without any retention time.
    #[test]
    fn the_tag_prefilter_keeps_tag_supported_candidates_without_rt() {
        let dir = scratch("tagpre");
        let (ms2, lib, _) = synthetic_run(&dir, 200, 60, false);
        let out = format!("{dir}/s.parquet");
        let sum = run(PrescreenParams {
            ms2: &ms2,
            library_precursors: &lib,
            run_windows: None,
            ms1: None,
            out: &out,
            config: &conf(cfg()),
            config_hash: "t",
            sample_candidates: 0,
            tags_only: true,
        })
        .unwrap();
        assert!(sum.cutoff.is_none() && sum.bypass_reason.is_none());
        let kept = survivors(&out);
        assert!(
            (0..60u32).all(|i| kept.contains(&i)),
            "planted ladders carry tags"
        );
        assert!(kept.len() < 200, "{}", kept.len());
        let rep: serde_json::Value =
            mumdia_io::json::read_json(&format!("{out}.report.json")).unwrap();
        assert_eq!(rep["stats"]["tags_only"], true);
        assert!(rep["stats"]["retrieved"].as_u64().unwrap() >= 60);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn both_scores(sp: &Spectra, pf: &str, c: &PrescreenConfig) -> (f64, f64) {
        let g = &sp.groups[0];
        let hist = Histogram::build((0..sp.rt.len()).map(|s| sp.peaks(s)), sp.nbin);
        let neglnp: Vec<f64> = sp
            .mz
            .iter()
            .map(|&m| -hist.probability(m as f64).ln())
            .collect();
        let cand = Candidate { nz: 2, rt: None };
        let m = residue_masses(pf).unwrap();
        score_in_group(
            &cand,
            &m,
            sp,
            g,
            None,
            &neglnp,
            c,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
    }

    /// Crowding divides a spectrum's score by (peaks / window mean)^exponent, so the same matched
    /// fragments score lower in a crowded spectrum; exponent 0 leaves the score unchanged.
    #[test]
    fn crowding_lowers_scores_from_crowded_spectra() {
        let frag = planted("PEPTIDEK", 2);
        let mut crowded = frag.clone();
        crowded.extend((0..400).map(|i| (1500.0 + i as f32 * 0.37, 5.0)));
        let sparse = with_background(vec![(100.0, frag.clone()), (200.0, crowded.clone())]);
        let only_crowded = with_background(vec![(200.0, crowded)]);
        let mut c = cfg();
        let plain = both_scores(&only_crowded, "PEPTIDEK", &c).0;
        c.crowding_exponent = 0.25;
        let adjusted = both_scores(&only_crowded, "PEPTIDEK", &c).0;
        assert!(adjusted < plain, "{adjusted} {plain}");
        // With a sparse copy present, the best spectrum is the sparse one.
        assert!(both_scores(&sparse, "PEPTIDEK", &c).0 > adjusted);
    }

    /// The repeat bonus needs the same fragments in a neighbouring scan within 8 s.
    #[test]
    fn repeat_support_needs_a_neighbouring_scan() {
        let frag = planted("PEPTIDEK", 2);
        let mut c = cfg();
        c.repeat_bonus = 0.1;
        let twice = with_background(vec![(100.0, frag.clone()), (104.0, frag.clone())]);
        let once = with_background(vec![(100.0, frag.clone())]);
        let far = with_background(vec![(100.0, frag.clone()), (120.0, frag)]);
        assert!(both_scores(&twice, "PEPTIDEK", &c).1 > 0.0);
        assert_eq!(both_scores(&once, "PEPTIDEK", &c).1, 0.0);
        assert_eq!(
            both_scores(&far, "PEPTIDEK", &c).1,
            0.0,
            "20 s apart is not a repeat"
        );
        c.repeat_bonus = 0.0;
        assert_eq!(both_scores(&twice, "PEPTIDEK", &c).1, 0.0, "off by default");
    }

    /// The occupancy bitmap only skips searches that cannot match: scores with and without it
    /// are identical, including peaks placed on the tolerance and bin edges.
    #[test]
    fn the_occupancy_bitmap_never_changes_a_score() {
        let mut st = 0x5EEDu64;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f64 / (1u64 << 53) as f64
        };
        let peps = [
            "PEPTIDEK",
            "LGEYGFQNALIVR",
            "GASPVTLK",
            "M[Oxidation]CNDEFK",
        ];
        for round in 0..20 {
            let mut spectra = Vec::new();
            for t in 0..12 {
                let mut p: Vec<(f32, f32)> = (0..150)
                    .map(|_| ((150.0 + next() * 1500.0) as f32, 1.0))
                    .collect();
                for pf in &peps {
                    for x in frags(pf, 2, false) {
                        let r = next();
                        // On the tolerance edge, on a bin edge, or near the fragment.
                        let m = if r < 0.2 {
                            x + 0.005 * if next() < 0.5 { 1.0 } else { -1.0 }
                        } else if r < 0.4 {
                            ((x * 100.0).floor() + 0.5) / 100.0
                        } else if r < 0.6 {
                            x + (next() - 0.5) * 0.009
                        } else {
                            continue;
                        };
                        p.push((m as f32, if next() < 0.1 { 0.0 } else { 3.0 }));
                    }
                }
                spectra.push((t as f64 * 5.0, p));
            }
            let sp = one_window(spectra);
            let g = &sp.groups[0];
            let hist = Histogram::build((0..sp.rt.len()).map(|s| sp.peaks(s)), sp.nbin);
            let occ = Occupancy::build(&sp, g, &hist);
            let neglnp: Vec<f64> = sp
                .mz
                .iter()
                .map(|&m| -hist.probability(m as f64).ln())
                .collect();
            let mut c = cfg();
            if round % 2 == 1 {
                c.frag_tol_da = 0.012;
                c.repeat_bonus = 0.1;
                c.crowding_exponent = 0.25;
            }
            for pf in &peps {
                let cand = Candidate { nz: 2, rt: None };
                let m = residue_masses(pf).unwrap();
                let run = |o: Option<&Occupancy>| {
                    score_in_group(
                        &cand,
                        &m,
                        &sp,
                        g,
                        o,
                        &neglnp,
                        &c,
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &mut Vec::new(),
                    )
                };
                assert_eq!(run(None), run(Some(&occ)), "{pf} round {round}");
            }
        }
    }

    /// Before prediction with both filters on, the survivors are the intersection of the tag
    /// prefilter's and the no-RT score's.
    #[test]
    fn both_prediction_prefilters_intersect() {
        let dir = scratch("bothpre");
        let (ms2, lib, _) = synthetic_run(&dir, 300, 90, false);
        let mut c = cfg();
        c.tag_prefilter = true;
        c.score_before_prediction = true;
        c.crowding_exponent = 0.25;
        let config = conf(c.clone());
        let both = run_before_prediction(&config, "t", &ms2, &lib, &dir)
            .unwrap()
            .unwrap();
        let tags = survivors(&format!("{dir}/tag_prefilter_survivors.parquet"));
        let score = survivors(&format!("{dir}/score_prefilter_survivors.parquet"));
        let want: Vec<u32> = tags.iter().copied().filter(|x| score.contains(x)).collect();
        assert_eq!(survivors(&both), want);
        assert!(
            (0..90u32).all(|i| want.contains(&i)),
            "planted candidates pass both"
        );
        c.tag_prefilter = false;
        c.score_before_prediction = false;
        assert!(run_before_prediction(&conf(c), "t", &ms2, &lib, &dir)
            .unwrap()
            .is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
