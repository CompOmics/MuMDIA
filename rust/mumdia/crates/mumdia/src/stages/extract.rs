//! Stage D `mumdia extract`: targeted 3D extraction (docs/09_extract.md).
//!
//! Data-driven and peak-major: observed peaks probe the inverted fragment index,
//! and a candidate hypothesis is materialized only where fragment evidence
//! exists (a sparse accumulator keyed by `candidate_id`, entries created on first
//! collision). Work scales with peak-candidate collisions, not library size.
//! RT is applied as a per-candidate window post-filter (the documented
//! fallback); MVP is 3D so IM is absent.
//!
//! The cascade: (a) isolation-window candidate range + RT window membership,
//! (b) cheap matched-fragment presence gate, (c) matched-fragment count + a
//! consecutive-scan co-elution run, (d) apex detection. Exact intensity scores
//! are computed in the features stage from the emitted chromatograms.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{ExtractConfig, GateMode, PeakClaim};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
use serde_json::json;
use tracing::{info, warn};

use mumdia_core::constants::{ppm_bounds, ISOTOPE_SPACING};

use crate::index::Library;
use crate::matchers::fragindex::{FragIndex, WindowNarrow};
use crate::spectra::{load_ms1, load_ms2, Ms1Scan};
use mumdia_core::config::MatcherKind;
use mumdia_core::types::Ms2Scan;
use rayon::prelude::*;

/// The matcher backend plus the values that never change across a probe: which
/// index is in use, the library, and the fragment tolerance. Bundling them keeps the
/// per-peak call down to what actually varies.
struct Prober<'a> {
    fidx: Option<&'a FragIndex>,
    lib: &'a Library,
    frag_tol: f64,
}

impl Prober<'_> {
    /// Probe one (offset-corrected) query m/z, invoking `f(candidate_id,
    /// candidate_local_fragment_ordinal, predicted_intensity)` for every verified
    /// match in the candidate window `[lo, hi)`. Bucketed resolves the fragment
    /// ordinal via `Library::local_frag_index` (nearest stored m/z); fragindex
    /// carries the true generating ordinal in `post_frag` (a semantic change for
    /// candidates with fragments at sub-f32-identical m/z, per the plan). Both apply
    /// the same tolerance; the fragindex index is already built at `frag_tol`.
    ///
    /// `nw` is an optional per-window narrowing cache for the fragindex path; it must
    /// have been built for this same `(lo, hi)`. Passing `None` is always correct and
    /// gives the uncached probe.
    #[inline]
    fn probe(
        &self,
        nw: Option<&mut WindowNarrow>,
        q_mz: f64,
        lo: u32,
        hi: u32,
        f: &mut dyn FnMut(u32, u16, f32),
    ) {
        match (self.fidx, nw) {
            (Some(idx), Some(nw)) => {
                idx.probe_peak_win(nw, q_mz, |cid, _pmz, pint, frag| f(cid, frag, pint))
            }
            (Some(idx), None) => {
                idx.probe_peak(q_mz, lo, hi, |cid, _pmz, pint, frag| f(cid, frag, pint))
            }
            (None, _) => {
                let lib = self.lib;
                lib.page_search(q_mz, self.frag_tol, lo, hi, |cid, frag_mz, pi| {
                    let frag = lib.local_frag_index(cid, frag_mz) as u16;
                    f(cid, frag, pi);
                })
            }
        }
    }
}

pub struct ExtractParams<'a> {
    pub ms2: &'a str,
    pub library_precursors: &'a str,
    pub library_fragments: &'a str,
    pub run_windows: &'a str,
    /// Optional MS1 spectra for precursor isotope-envelope features
    /// (docs/10_features.md). When absent, MS1 columns are null.
    pub ms1: Option<&'a str>,
    /// Optional per-run mass recalibration (search-seed `<seed>.masscal.json`):
    /// systematic fragment ppm offset + learned tolerance.
    pub mass_cal: Option<&'a str>,
    pub out_psms: &'a str,
    pub out_chrom: &'a str,
    /// Optional candidate allowlist (a prior run's `psms.parquet`): restrict
    /// extraction to these `candidate_id`s. Used for "gate first, then compete":
    /// run a cheap gate-on pass, then re-extract with a peak-claim strategy over
    /// only the accepted survivors, so the expensive two-pass profile map is built
    /// over ~10^5 candidates instead of ~10^7.
    pub restrict_candidates: Option<&'a str>,
    pub cfg: &'a ExtractConfig,
    pub config_hash: &'a str,
}

/// One observed hit: scan RT, candidate-local fragment index, observed intensity
/// and observed m/z (for mass-accuracy features).
struct Hit {
    rt: f64,
    frag: u16,
    inten: f32,
    obs_mz: f64,
}

type ChromOutputRow = (u32, String, f64, f64, f32, Vec<f32>, Vec<f32>);

/// One observed peak for the per-scan demix: (observed intensity, observed m/z, claimants
/// as (candidate_id, fragment_ordinal, predicted_intensity)).
type DemixRow = (f32, f64, Vec<(u32, u16, f32)>);

/// Per-candidate contested-peak statistics from the co-elution arbitration
/// (two-pass path). `won`/`lost` are the summed observed intensity of shared peaks
/// this candidate won (was the most-eluting claimant) or lost to a better
/// co-eluter; `n_won`/`n_lost` are the corresponding peak-instance counts; and
/// `apportioned` is the candidate's co-elution-weighted proportional share of the
/// contested intensity (what it would keep under `CoelutionProportional`). These
/// feed the soft competition features (`contested_frac`, `contested_count_frac`,
/// `apportioned_frac`) without removing any candidate.
#[derive(Default, Clone, Copy)]
struct Contested {
    won: f64,
    lost: f64,
    n_won: u32,
    n_lost: u32,
    apportioned: f64,
}

/// Index of the value in ascending `rts` nearest to `t` (binary search).
fn nearest_index(rts: &[f64], t: f64) -> usize {
    if rts.is_empty() {
        return 0;
    }
    let p = rts.partition_point(|&r| r < t);
    if p == 0 {
        0
    } else if p >= rts.len() {
        rts.len() - 1
    } else if (t - rts[p - 1]).abs() <= (rts[p] - t).abs() {
        p - 1
    } else {
        p
    }
}

/// Composite per-claimant weight-cue multiplier for `PeakClaim::CoelutionMultiCue`
/// (modular fragment-competition framework). Each enabled [`ClaimCues`] cue contributes
/// a factor in [0,1]; disabled cues contribute 1.0, so the product is 1.0 when no cue
/// is on and the arbitration reduces exactly to the elution-profile-height weight.
/// Label-blind (reads only observed/predicted fragment m/z), so target/decoy
/// exchangeability is preserved.
#[inline]
#[allow(clippy::too_many_arguments)]
fn claim_cue_multiplier(
    cfg: &ExtractConfig,
    lib: &Library,
    cid: u32,
    frag: u16,
    obs_mz: f64,
    rt: f64,
    rt_cal: &[f64],
    ms1_scans: &[Ms1Scan],
    ms1_rts: &[f64],
) -> f32 {
    let mut w = 1.0f32;
    if cfg.claim_cues.mz_close {
        // Sub-tolerance m/z proximity (S3): the observed peak sits at the true owner's
        // m/z, so a claimant whose predicted fragment m/z is closer wins more weight.
        let (mzs, _, _) = lib.cand_frags(cid);
        let pred_mz = mzs.get(frag as usize).copied().unwrap_or(obs_mz);
        let ppm = mumdia_core::constants::ppm_diff(obs_mz, pred_mz) as f32;
        let sigma = (cfg.claim_cues.mz_close_sigma_ppm as f32).max(1e-6);
        w *= (-(ppm / sigma).powi(2)).exp();
    }
    if cfg.claim_cues.rt_prior {
        // DeepLC RT prior (S3): down-weight a claimant whose calibrated predicted RT is
        // far from the current scan (a briefly-co-eluting interferent). No-op when the
        // predicted RT is unset (0).
        let rt_pred = rt_cal.get(cid as usize).copied().unwrap_or(0.0);
        if rt_pred > 0.0 {
            let tau = (cfg.claim_cues.rt_prior_tau_s as f32).max(1e-3);
            let d = (rt - rt_pred) as f32;
            w *= (-(d * d) / (2.0 * tau * tau)).exp();
        }
    }
    if cfg.claim_cues.ms1_support && !ms1_scans.is_empty() {
        // MS1 precursor-envelope support (S4): the claimant's OWN precursor isotope
        // envelope at the nearest MS1 scan. Absent mono precursor -> down-weight; a
        // present mono with an implausible +1/mono ratio -> mild down-weight. A decoy's
        // precursor m/z is well-defined but has no real co-eluting MS1 signal.
        let cand = &lib.cands[cid as usize];
        let j = nearest_index(ms1_rts, rt);
        let s = &ms1_scans[j];
        let z = (cand.charge.max(1)) as f64;
        let sp = ISOTOPE_SPACING / z;
        let tol = cfg.prec_tol_ppm;
        let mono = sum_near(&s.mz, &s.intensity, cand.precursor_mz, tol);
        if mono <= 0.0 {
            w *= 0.5;
        } else {
            let i1 = sum_near(&s.mz, &s.intensity, cand.precursor_mz + sp, tol);
            let ratio = i1 / mono;
            if !(0.05..=2.0).contains(&ratio) {
                w *= 0.75;
            }
        }
    }
    w
}

/// Everything the spectrum-centric NNLS demix (D2, fragment-competition report) derives
/// from an apex SCAN alone: the co-isolated candidate x fragment-channel design matrix
/// (rows = the scan's observed peaks, columns = candidates that claim a peak,
/// `A[peak,cand]` = the candidate's predicted intensity for its matching fragment), the
/// observed vector, the NNLS solution of `min_{beta>=0} ||A beta - y||^2`, and the
/// quantities computed from those that do not depend on which candidate is asked about.
///
/// This used to be recomputed per CANDIDATE even though the candidate id only selects a
/// column and gates an early return, so every candidate re-probed every peak of its apex
/// scan and re-ran the NNLS. Solving once per scan and reading each candidate's column out
/// is exactly equivalent: nothing above the column read depends on the candidate. The
/// apex-scan lookup keys on the candidate's `apex_rt`, and the scan it resolves to has
/// `scan.rt_seconds == apex_rt` by construction, so the RT admission test is
/// scan-determined too.
///
/// Deterministic: peaks in scan order, candidate columns in sorted `cid` order, ordered
/// reductions.
struct DemixScan {
    /// candidate id -> design-matrix column, ascending by id.
    col_of: std::collections::BTreeMap<u32, usize>,
    /// Row-major `m x n` design matrix.
    a: Vec<f64>,
    /// Observed intensity per row.
    y: Vec<f64>,
    m: usize,
    n: usize,
    /// NNLS solution, one coefficient per column.
    beta: Vec<f64>,
    sum_beta: f64,
    /// Joint residual-explained fraction `1 - ||y - A beta||^2 / ||y||^2`.
    explained: f64,
    /// Per-column abundance seeded from the channels that column claims ALONE (its unique
    /// ions), NaN where it has none. Feeds the D1 shadow subtraction.
    a_p: Vec<f64>,
}

/// The acquisition scan at `apex_rt` whose isolation window covers `prec_mz`, if any.
/// Two candidates sharing an apex RT but sitting in different windows resolve to
/// different scans, so this index -- not the RT -- is what a solved problem is keyed by.
fn demix_apex_scan(
    scans: &[Ms2Scan],
    rt_scan: &HashMap<u64, Vec<u32>>,
    apex_rt: f64,
    prec_mz: f64,
) -> Option<usize> {
    rt_scan.get(&apex_rt.to_bits()).and_then(|v| {
        v.iter()
            .copied()
            .find(|&s| {
                let w = &scans[s as usize].window;
                w.lower_mz <= prec_mz && prec_mz <= w.upper_mz
            })
            .map(|s| s as usize)
    })
}

/// Assemble and solve the demix problem for one apex scan. `None` when the scan yields no
/// usable channels or columns.
#[allow(clippy::too_many_arguments)]
fn demix_solve_scan(
    idx: Option<&FragIndex>,
    lib: &Library,
    scan: &Ms2Scan,
    mass_off: &MassOffset,
    frag_tol: f64,
    apex_rt: f64,
    rt_lo: &[f64],
    rt_hi: &[f64],
    cfg: &ExtractConfig,
) -> Option<DemixScan> {
    let pr = Prober {
        fidx: idx,
        lib,
        frag_tol,
    };
    let (lo, hi) = lib.candidate_range(scan.window.lower_mz, scan.window.upper_mz);
    // Column set (candidates), deterministic by sorted cid. Rows carry (obs, claimants).
    let mut col_of: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    let mut rows: Vec<(f64, Vec<(u32, f32)>)> = Vec::new();
    let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
    for peak in &scan.peaks {
        let q_mz = peak.mz / mass_off.factor_at(peak.mz);
        claimants.clear();
        {
            let mut push = |c: u32, frag: u16, pi: f32| {
                let cc = c as usize;
                if apex_rt < rt_lo[cc] || apex_rt > rt_hi[cc] {
                    return;
                }
                claimants.push((c, frag, pi));
            };
            pr.probe(None, q_mz, lo, hi, &mut push);
        }
        if claimants.is_empty() {
            continue;
        }
        let mut entry: Vec<(u32, f32)> = Vec::with_capacity(claimants.len());
        for &(c, _f, pi) in &claimants {
            if col_of.len() < cfg.demix_max_candidates || col_of.contains_key(&c) {
                col_of.entry(c).or_insert(0);
                entry.push((c, pi));
            }
        }
        if !entry.is_empty() {
            rows.push((peak.intensity as f64, entry));
        }
    }
    let n = col_of.len();
    let m = rows.len();
    if n == 0 || m == 0 {
        return None;
    }
    for (k, v) in col_of.values_mut().enumerate() {
        *v = k;
    }
    let mut a = vec![0.0f64; m * n];
    let mut y = vec![0.0f64; m];
    for (r, (obs, ents)) in rows.iter().enumerate() {
        y[r] = *obs;
        for &(c, pi) in ents {
            if let Some(&col) = col_of.get(&c) {
                // A candidate matching a peak via >1 fragment keeps its largest predicted
                // intensity for that channel (deterministic, order-independent).
                let cell = &mut a[r * n + col];
                if (pi as f64) > *cell {
                    *cell = pi as f64;
                }
            }
        }
    }
    let lambda = cfg.demix_lambda.max(1e-9);
    let beta = crate::solve::nnls(&a, m, n, &y, lambda, 200 * n.max(1));
    let sum_beta: f64 = beta.iter().sum();
    // Explained fraction = 1 - ||y - A beta||^2 / ||y||^2.
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (r, &yr) in y.iter().enumerate() {
        let mut pred = 0.0;
        for col in 0..n {
            pred += a[r * n + col] * beta[col];
        }
        let d = yr - pred;
        num += d * d;
        den += yr * yr;
    }
    let explained = if den > 0.0 {
        (1.0 - num / den).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Per-column abundance from the rows it claims ALONE: collect the unique-row ratios per
    // column, then take the median deterministically.
    let mut a_p = vec![f64::NAN; n];
    {
        let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); n];
        for r in 0..m {
            let mut nz = 0usize;
            let mut col = 0usize;
            for j in 0..n {
                if a[r * n + j] > 0.0 {
                    nz += 1;
                    col = j;
                }
            }
            if nz == 1 && a[r * n + col] > 0.0 {
                ratios[col].push(y[r] / a[r * n + col]);
            }
        }
        for (j, rs) in ratios.iter_mut().enumerate() {
            if !rs.is_empty() {
                rs.sort_by(|x, z| x.total_cmp(z));
                a_p[j] = rs[rs.len() / 2];
            }
        }
    }
    Some(DemixScan {
        col_of,
        a,
        y,
        m,
        n,
        beta,
        sum_beta,
        explained,
        a_p,
    })
}

/// `(deconv_explained_frac, deconv_active, deconv_share, deconv_max_collinearity,
/// shadow_kept_frac)` for one candidate.
type DemixFeatures = (f64, f64, f64, f64, f64);

/// Read candidate `cid`'s demix features out of its apex scan's solved problem:
/// `(deconv_explained_frac, deconv_active, deconv_share, deconv_max_collinearity,
/// shadow_kept_frac)`. Zeros when the candidate is not one of the scan's columns.
fn demix_features_for(d: &DemixScan, cid: u32) -> DemixFeatures {
    let c_col = match d.col_of.get(&cid) {
        Some(&c) => c,
        None => return (0.0, 0.0, 0.0, 0.0, 0.0),
    };
    let (a, y, m, n) = (&d.a, &d.y, d.m, d.n);
    let coef = d.beta[c_col].max(0.0);
    let share = if d.sum_beta > 0.0 {
        coef / d.sum_beta
    } else {
        0.0
    };
    let active = if coef > 1e-6 { 1.0 } else { 0.0 };
    // D1 shadow-spectrum: subtract every OTHER candidate's unique-ion-seeded contribution
    // from candidate c's channels and measure how much of c's observed intensity survives.
    // A real second peptide keeps most of its signal; a pure borrower is subtracted away.
    // This sidesteps the shared-peak circularity by seeding abundances from unique ions
    // only. 1.0 (kept all) when c has no interfering neighbor with unique ions.
    let shadow_kept = {
        let (mut kept, mut total) = (0.0f64, 0.0f64);
        for r in 0..m {
            let dc = a[r * n + c_col];
            if dc <= 0.0 {
                continue;
            }
            total += y[r];
            let mut sub = 0.0;
            for j in 0..n {
                if j != c_col && d.a_p[j].is_finite() {
                    sub += d.a_p[j] * a[r * n + j];
                }
            }
            kept += (y[r] - sub).max(0.0);
        }
        if total > 0.0 {
            (kept / total).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    // Identifiability (D3): the maximum cosine similarity of candidate c's design column
    // with any other column. Near 1 = c is near-degenerate with a rival, so its
    // coefficient is an essentially arbitrary split (distrust the demix). No incumbent
    // engine emits this. O(m n), reuses the assembled matrix.
    let norm_c = {
        let mut s = 0.0;
        for r in 0..m {
            let v = a[r * n + c_col];
            s += v * v;
        }
        s.sqrt()
    };
    let mut max_collin = 0.0f64;
    if norm_c > 0.0 {
        for j in 0..n {
            if j == c_col {
                continue;
            }
            let (mut dot, mut nj) = (0.0f64, 0.0f64);
            for r in 0..m {
                let vc = a[r * n + c_col];
                let vj = a[r * n + j];
                dot += vc * vj;
                nj += vj * vj;
            }
            let nj = nj.sqrt();
            if nj > 0.0 {
                max_collin = max_collin.max(dot / (norm_c * nj));
            }
        }
    }
    (
        d.explained,
        active,
        share,
        max_collin.clamp(0.0, 1.0),
        shadow_kept,
    )
}

/// Sum intensities of peaks within `tol_ppm` of `target` (m/z-sorted arrays).
fn sum_near(mz: &[f64], inten: &[f32], target: f64, tol_ppm: f64) -> f32 {
    if mz.is_empty() {
        return 0.0;
    }
    let (lo, hi) = ppm_bounds(target, tol_ppm);
    let s = mz.partition_point(|&m| m < lo);
    let mut acc = 0.0f32;
    let mut i = s;
    while i < mz.len() && mz[i] <= hi {
        acc += inten[i];
        i += 1;
    }
    acc
}

/// Co-elution acceptance score (sensitivity program): predicted-intensity-weighted
/// mean Pearson correlation of each matched fragment's XIC to the signature-ion
/// reference profile, over the elution scan groups. High when the peptide's own
/// fragments co-elute (real); low when a matched fragment is a non-co-eluting
/// interferent that only coincides at the apex. More robust to chimeric DIA
/// interference than the single-scan apex intensity Pearson. Returns 1.0 (do not
/// reject) when there are too few scan groups or no reference signal.
/// Contiguous elution-peak scan indices `[lo, hi]` around the signature-ion apex
/// (scans above 10% of the reference apex height) plus the reference profile.
/// `None` when there are too few scans, no reference signal, or a < 3-scan peak.
/// Over the full (wide) extraction window the traces are mostly zeros and any
/// correlation is noise; the spectral/co-elution gates are only meaningful across
/// the elution peak itself.
fn peak_window(
    groups: &[(f64, std::collections::BTreeMap<u16, f32>)],
    sig: &[u16],
) -> Option<(usize, usize, Vec<f64>)> {
    if groups.len() < 3 {
        return None;
    }
    let refp: Vec<f64> = groups
        .iter()
        .map(|(_, m)| {
            sig.iter()
                .map(|o| *m.get(o).unwrap_or(&0.0) as f64)
                .sum::<f64>()
        })
        .collect();
    let (apex, apex_v) =
        refp.iter().enumerate().fold(
            (0usize, 0.0f64),
            |(bi, bv), (i, v)| if *v > bv { (i, *v) } else { (bi, bv) },
        );
    if apex_v <= 0.0 {
        return None;
    }
    let thr = 0.1 * apex_v;
    let (mut lo, mut hi) = (apex, apex);
    while lo > 0 && refp[lo - 1] >= thr {
        lo -= 1;
    }
    while hi + 1 < refp.len() && refp[hi + 1] >= thr {
        hi += 1;
    }
    if hi - lo + 1 < 3 {
        return None;
    }
    Some((lo, hi, refp))
}

/// Peak-integrated spectral Pearson: correlate the PEAK-SUMMED observed spectrum
/// (each predicted fragment integrated over the elution-peak scans) with the
/// predicted intensities. Averaging over the peak removes the single-interfered-
/// scan fragility of the apex-only Pearson. Returns 1.0 when no peak is resolved.
fn peak_spectral_score(
    groups: &[(f64, std::collections::BTreeMap<u16, f32>)],
    sig: &[u16],
    fints0: &[f32],
) -> f64 {
    let (lo, hi, _refp) = match peak_window(groups, sig) {
        Some(w) => w,
        None => return 1.0,
    };
    let obs: Vec<f64> = (0..fints0.len())
        .map(|f| {
            groups[lo..=hi]
                .iter()
                .map(|(_, m)| *m.get(&(f as u16)).unwrap_or(&0.0) as f64)
                .sum::<f64>()
        })
        .collect();
    let pred: Vec<f64> = fints0.iter().map(|x| *x as f64).collect();
    crate::stats::pearson(&obs, &pred)
}

/// Co-elution acceptance score (temporal): predicted-intensity-weighted mean
/// Pearson correlation of each matched fragment's XIC to the signature-ion
/// reference profile, over the elution peak. High when the peptide's own fragments
/// co-elute; low when a matched fragment only coincides at the apex. Orthogonal to
/// the intensity-agreement of `peak_spectral_score`.
fn coelution_gate_score(
    groups: &[(f64, std::collections::BTreeMap<u16, f32>)],
    distinct: &[u16],
    sig: &[u16],
    fints0: &[f32],
) -> f64 {
    let (lo, hi, refp) = match peak_window(groups, sig) {
        Some(w) => w,
        None => return 1.0,
    };
    let refw = &refp[lo..=hi];
    let (mut wsum, mut wtot) = (0.0f64, 0.0f64);
    for &f in distinct {
        let tr: Vec<f64> = groups[lo..=hi]
            .iter()
            .map(|(_, m)| *m.get(&f).unwrap_or(&0.0) as f64)
            .collect();
        if tr.iter().any(|x| *x > 0.0) {
            let c = crate::stats::pearson(&tr, refw).max(0.0);
            let w = *fints0.get(f as usize).unwrap_or(&0.0) as f64 + 1e-9;
            wsum += c * w;
            wtot += w;
        }
    }
    if wtot > 0.0 {
        wsum / wtot
    } else {
        1.0
    }
}

/// fragindex non-two-pass accumulation over isolation-window groups, in parallel.
/// Each scan belongs to exactly one window, so the groups are independent; the
/// per-candidate hit lists are merged by concatenating in (window-sorted) group
/// order. Bit-identical to serial accumulation: the per-candidate cascade rt-sorts
/// hits before the apex sum, and same-rt hits for a candidate all come from one
/// window, so the concatenation order does not affect the rt-sorted result. Only
/// the PeakClaim::None / Winner / Proportional (non-two-pass) strategies use this;
/// the co-elution two-pass path stays serial.
/// Per-peak mass-offset correction applied to an observed peak m/z before library
/// matching. Either a single scalar ppm offset (`grid_*` empty, the default) or an
/// m/z-dependent grid (sorted ascending) that is linearly interpolated and clamped
/// at the ends. `factor_at(mz)` returns the divisor `1 + ppm(mz) * 1e-6`.
struct MassOffset {
    scalar_ppm: f64,
    grid_mz: Vec<f64>,
    grid_ppm: Vec<f64>,
}
impl MassOffset {
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

fn extract_accumulate_windows(
    idx: &FragIndex,
    scans: &[Ms2Scan],
    rt_lo: &[f64],
    rt_hi: &[f64],
    mass_off: &MassOffset,
    cfg: &ExtractConfig,
) -> HashMap<u32, Vec<Hit>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (si, scan) in scans.iter().enumerate() {
        groups
            .entry((
                scan.window.lower_mz.to_bits(),
                scan.window.upper_mz.to_bits(),
            ))
            .or_default()
            .push(si);
    }
    let group_vec: Vec<Vec<usize>> = groups.into_values().collect();

    let partials: Vec<Vec<(u32, Vec<Hit>)>> = group_vec
        .par_iter()
        .map(|ids| {
            if ids.is_empty() {
                return Vec::new();
            }
            let w = &scans[ids[0]].window;
            let (lo, hi) = idx.candidate_range(w.lower_mz, w.upper_mz);
            if hi <= lo {
                return Vec::new();
            }
            let mut local: HashMap<u32, Vec<Hit>> = HashMap::new();
            let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
            // `(lo, hi)` is fixed for this whole isolation window and every scan of it
            // reprobes the same bins, so cache each bin's narrowed posting range once
            // instead of binary-searching it per peak.
            let mut nw = idx.window_narrow(lo, hi);
            for &si in ids {
                let scan = &scans[si];
                let rt = scan.rt_seconds;
                for peak in &scan.peaks {
                    let inten = peak.intensity;
                    let q_mz = peak.mz / mass_off.factor_at(peak.mz);
                    let obs_mz = peak.mz;
                    claimants.clear();
                    idx.probe_peak_win(&mut nw, q_mz, |cid, _pmz, pint, frag| {
                        let c = cid as usize;
                        if rt < rt_lo[c] || rt > rt_hi[c] {
                            return;
                        }
                        claimants.push((cid, frag, pint));
                    });
                    if claimants.is_empty() {
                        continue;
                    }
                    match cfg.peak_claim {
                        PeakClaim::WinnerPredictedIntensity => {
                            let mut best = 0usize;
                            for i in 1..claimants.len() {
                                let a = claimants[i];
                                let b = claimants[best];
                                if a.2 > b.2 || (a.2 == b.2 && a.0 < b.0) {
                                    best = i;
                                }
                            }
                            let (cid, frag, _) = claimants[best];
                            local.entry(cid).or_default().push(Hit {
                                rt,
                                frag,
                                inten,
                                obs_mz,
                            });
                        }
                        PeakClaim::Proportional => {
                            let sump: f32 = claimants.iter().map(|c| c.2.max(0.0)).sum();
                            for &(cid, frag, pi) in &claimants {
                                let share = if sump > 0.0 {
                                    inten * (pi.max(0.0) / sump)
                                } else {
                                    inten / claimants.len() as f32
                                };
                                local.entry(cid).or_default().push(Hit {
                                    rt,
                                    frag,
                                    inten: share,
                                    obs_mz,
                                });
                            }
                        }
                        _ => {
                            for &(cid, frag, _) in &claimants {
                                local.entry(cid).or_default().push(Hit {
                                    rt,
                                    frag,
                                    inten,
                                    obs_mz,
                                });
                            }
                        }
                    }
                }
            }
            local.into_iter().collect()
        })
        .collect();

    let mut acc: HashMap<u32, Vec<Hit>> = HashMap::new();
    for part in partials {
        for (cid, hits) in part {
            acc.entry(cid).or_default().extend(hits);
        }
    }
    acc
}

/// Parallel two-pass co-elution peak-claim. Each isolation-window group is
/// processed independently (a candidate's precursor m/z places it in one window,
/// and a peak's claimants come only from that window via `candidate_range`), so
/// both the base accumulation (pass 1, for elution profiles) and the arbitration
/// (pass 2) fan out across the ~150 windows. Returns the (possibly reassigned)
/// accumulation and per-candidate (won, lost) contested intensity. Mirrors the
/// serial two-pass exactly, window-partitioned; merge is disjoint across windows
/// (extend/sum is overlap-safe if windows ever overlap in m/z).
#[allow(clippy::too_many_arguments)]
fn extract_twopass_windows(
    idx: Option<&FragIndex>,
    lib: &Library,
    scans: &[Ms2Scan],
    rt_lo: &[f64],
    rt_hi: &[f64],
    rt_cal: &[f64],
    ms1_scans: &[Ms1Scan],
    ms1_rts: &[f64],
    mass_off: &MassOffset,
    frag_tol: f64,
    cfg: &ExtractConfig,
    restrict: Option<&std::collections::HashSet<u32>>,
    reassign: bool,
    claim_margin: f32,
) -> (HashMap<u32, Vec<Hit>>, HashMap<u32, Contested>) {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (si, scan) in scans.iter().enumerate() {
        groups
            .entry((
                scan.window.lower_mz.to_bits(),
                scan.window.upper_mz.to_bits(),
            ))
            .or_default()
            .push(si);
    }
    let group_vec: Vec<Vec<usize>> = groups.into_values().collect();
    let pr = Prober {
        fidx: idx,
        lib,
        frag_tol,
    };

    type Part = (Vec<(u32, Vec<Hit>)>, Vec<(u32, Contested)>);
    let partials: Vec<Part> = group_vec
        .par_iter()
        .map(|ids| {
            if ids.is_empty() {
                return (Vec::new(), Vec::new());
            }
            let w = &scans[ids[0]].window;
            let (lo, hi) = lib.candidate_range(w.lower_mz, w.upper_mz);
            if hi <= lo {
                return (Vec::new(), Vec::new());
            }
            let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
            // `(lo, hi)` is fixed for this whole isolation window and both passes
            // reprobe the same bins across every scan of it, so narrow each bin once.
            let mut nw = idx.map(|i| i.window_narrow(lo, hi));
            // PASS 1: base accumulation (full peak intensity) for elution profiles.
            let mut acc1: HashMap<u32, Vec<Hit>> = HashMap::new();
            for &si in ids {
                let scan = &scans[si];
                let rt = scan.rt_seconds;
                for peak in &scan.peaks {
                    let inten = peak.intensity;
                    let q_mz = peak.mz / mass_off.factor_at(peak.mz);
                    let obs_mz = peak.mz;
                    claimants.clear();
                    {
                        let mut push = |cid: u32, frag: u16, pi: f32| {
                            let c = cid as usize;
                            if rt < rt_lo[c] || rt > rt_hi[c] {
                                return;
                            }
                            if let Some(s) = restrict {
                                if !s.contains(&cid) {
                                    return;
                                }
                            }
                            claimants.push((cid, frag, pi));
                        };
                        pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                    }
                    for &(cid, frag, _) in &claimants {
                        acc1.entry(cid).or_default().push(Hit {
                            rt,
                            frag,
                            inten,
                            obs_mz,
                        });
                    }
                }
            }
            let mut profile: HashMap<u32, HashMap<u64, f32>> = HashMap::new();
            for (cid, hits) in &acc1 {
                let m = profile.entry(*cid).or_default();
                for h in hits {
                    *m.entry(h.rt.to_bits()).or_insert(0.0) += h.inten;
                }
            }
            // S2 uniqueness-seeded EM: re-seed each candidate's elution profile from its
            // cue-weighted APPORTIONED intensity (not the full peak) for a fixed number of
            // iterations, so a borrowing candidate's profile is no longer inflated by the
            // peaks it borrows. A single-claimant (uncontested) peak contributes its full
            // intensity every iteration -> an immovable anchor. Deterministic (fixed N,
            // ordered f32 reductions). Only under CoelutionMultiCue; 0 iters = no-op.
            let em_iters = if matches!(cfg.peak_claim, PeakClaim::CoelutionMultiCue) {
                cfg.claim_cues.apportion_em_iters
            } else {
                0
            };
            for _ in 0..em_iters {
                let mut next: HashMap<u32, HashMap<u64, f32>> = HashMap::new();
                for &si in ids {
                    let scan = &scans[si];
                    let rt = scan.rt_seconds;
                    let rtb = rt.to_bits();
                    for peak in &scan.peaks {
                        let inten = peak.intensity;
                        let q_mz = peak.mz / mass_off.factor_at(peak.mz);
                        let obs_mz = peak.mz;
                        claimants.clear();
                        {
                            let mut push = |cid: u32, frag: u16, pi: f32| {
                                let c = cid as usize;
                                if rt < rt_lo[c] || rt > rt_hi[c] {
                                    return;
                                }
                                if let Some(s) = restrict {
                                    if !s.contains(&cid) {
                                        return;
                                    }
                                }
                                claimants.push((cid, frag, pi));
                            };
                            pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                        }
                        if claimants.is_empty() {
                            continue;
                        }
                        let weights: Vec<f32> = claimants
                            .iter()
                            .map(|&(cid, frag, _)| {
                                let h = profile
                                    .get(&cid)
                                    .and_then(|m| m.get(&rtb))
                                    .copied()
                                    .unwrap_or(0.0);
                                if h > 0.0 {
                                    h * claim_cue_multiplier(
                                        cfg, lib, cid, frag, obs_mz, rt, rt_cal, ms1_scans, ms1_rts,
                                    )
                                } else {
                                    h
                                }
                            })
                            .collect();
                        let sum_w: f32 = weights.iter().copied().sum();
                        for (i, &(cid, _, _)) in claimants.iter().enumerate() {
                            let share = if sum_w > 0.0 {
                                inten * (weights[i] / sum_w)
                            } else {
                                inten / claimants.len() as f32
                            };
                            *next.entry(cid).or_default().entry(rtb).or_insert(0.0) += share;
                        }
                    }
                }
                profile = next;
            }
            // PASS 2: arbitrate each shared peak by which claimant is most eluting.
            let mut acc2: HashMap<u32, Vec<Hit>> = HashMap::new();
            let mut contested: HashMap<u32, Contested> = HashMap::new();
            let demix_mode = matches!(cfg.peak_claim, PeakClaim::CoelutionDemix);
            let shadow_mode = matches!(cfg.peak_claim, PeakClaim::CoelutionShadow);
            let demix_stride = cfg.demix_scan_stride.max(1);
            let mut demix_abund: BTreeMap<u32, f64> = BTreeMap::new();
            let mut demix_ctr = 0usize;
            for &si in ids {
                let scan = &scans[si];
                let rt = scan.rt_seconds;
                let rtb = rt.to_bits();
                // Spectrum-centric demix redistribution (CoelutionDemix): solve one NNLS
                // over this scan's co-isolated candidate x fragment matrix and split each
                // shared peak by beta_c * D[peak,c] (smooth joint deconvolution) rather than
                // the per-peak profile-height arbitration below.
                if demix_mode {
                    let mut cand: std::collections::BTreeSet<u32> =
                        std::collections::BTreeSet::new();
                    let mut prows: Vec<DemixRow> = Vec::new();
                    for peak in &scan.peaks {
                        let q_mz = peak.mz / mass_off.factor_at(peak.mz);
                        claimants.clear();
                        {
                            let mut push = |cid: u32, frag: u16, pi: f32| {
                                let c = cid as usize;
                                if rt < rt_lo[c] || rt > rt_hi[c] {
                                    return;
                                }
                                if let Some(s) = restrict {
                                    if !s.contains(&cid) {
                                        return;
                                    }
                                }
                                claimants.push((cid, frag, pi));
                            };
                            pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                        }
                        if claimants.is_empty() {
                            continue;
                        }
                        for &(cid, _, _) in &claimants {
                            cand.insert(cid);
                        }
                        prows.push((peak.intensity, peak.mz, claimants.clone()));
                    }
                    if prows.is_empty() {
                        continue;
                    }
                    // Solve the NNLS every `demix_stride` scans or whenever a new candidate
                    // enters the co-isolated set; reuse the stored abundances otherwise. The
                    // abundances change slowly across a few scans (elution is gradual), while
                    // each scan's own peaks + predicted intensities still drive the split.
                    let new_cand = cand.iter().any(|c| !demix_abund.contains_key(c));
                    if demix_abund.is_empty() || demix_ctr.is_multiple_of(demix_stride) || new_cand
                    {
                        let cols: Vec<u32> = cand.iter().copied().collect();
                        let n = cols.len();
                        let col_of: BTreeMap<u32, usize> =
                            cols.iter().enumerate().map(|(i, &c)| (c, i)).collect();
                        let m = prows.len();
                        let mut amat = vec![0.0f64; m * n];
                        let mut yv = vec![0.0f64; m];
                        for (r, (obs, _, cl)) in prows.iter().enumerate() {
                            yv[r] = *obs as f64;
                            for &(cid, _, pi) in cl {
                                let cell = &mut amat[r * n + col_of[&cid]];
                                if (pi as f64) > *cell {
                                    *cell = pi as f64;
                                }
                            }
                        }
                        let beta = crate::solve::nnls(
                            &amat,
                            m,
                            n,
                            &yv,
                            cfg.demix_lambda.max(1e-9),
                            200 * n.max(1),
                        );
                        demix_abund.clear();
                        for (&c, &col) in &col_of {
                            demix_abund.insert(c, beta[col]);
                        }
                    }
                    demix_ctr += 1;
                    // Apportion each peak by abundance_c * predicted_c (the joint split).
                    for (obs, obs_mz, cl) in &prows {
                        let mut denom = 0.0f64;
                        let mut winner = cl[0].0;
                        let mut best_bd = f64::NEG_INFINITY;
                        for &(cid, _, pi) in cl {
                            let bd = demix_abund.get(&cid).copied().unwrap_or(0.0) * pi as f64;
                            denom += bd;
                            if bd > best_bd || (bd == best_bd && cid < winner) {
                                best_bd = bd;
                                winner = cid;
                            }
                        }
                        for &(cid, frag, pi) in cl {
                            let bd = demix_abund.get(&cid).copied().unwrap_or(0.0) * pi as f64;
                            let share = if denom > 0.0 {
                                *obs as f64 * (bd / denom)
                            } else {
                                *obs as f64 / cl.len() as f64
                            };
                            let e = contested.entry(cid).or_default();
                            e.apportioned += share;
                            if cid == winner {
                                e.won += *obs as f64;
                                e.n_won += 1;
                            } else {
                                e.lost += *obs as f64;
                                e.n_lost += 1;
                            }
                            acc2.entry(cid).or_default().push(Hit {
                                rt,
                                frag,
                                inten: share as f32,
                                obs_mz: *obs_mz,
                            });
                        }
                    }
                    continue;
                }
                if shadow_mode {
                    // Shadow subtraction: estimate each candidate's abundance from its UNIQUE
                    // channels, then clean every candidate's channels by subtracting the other
                    // claimants' estimated contributions. No solve; several real co-eluters can
                    // both keep signal at a shared peak.
                    let mut prows: Vec<DemixRow> = Vec::new();
                    for peak in &scan.peaks {
                        let q_mz = peak.mz / mass_off.factor_at(peak.mz);
                        claimants.clear();
                        {
                            let mut push = |cid: u32, frag: u16, pi: f32| {
                                let c = cid as usize;
                                if rt < rt_lo[c] || rt > rt_hi[c] {
                                    return;
                                }
                                if let Some(s) = restrict {
                                    if !s.contains(&cid) {
                                        return;
                                    }
                                }
                                claimants.push((cid, frag, pi));
                            };
                            pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                        }
                        if claimants.is_empty() {
                            continue;
                        }
                        prows.push((peak.intensity, peak.mz, claimants.clone()));
                    }
                    if prows.is_empty() {
                        continue;
                    }
                    // Per-candidate abundance from unique (single-claimant) channels, median of
                    // observed/predicted. Deterministic (sorted cid, median of sorted ratios).
                    let mut uniq: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
                    for (obs, _, cl) in &prows {
                        if cl.len() == 1 {
                            let (cid, _, pi) = cl[0];
                            if pi > 0.0 {
                                uniq.entry(cid).or_default().push(*obs as f64 / pi as f64);
                            }
                        }
                    }
                    let mut a_p: BTreeMap<u32, f64> = BTreeMap::new();
                    for (cid, mut v) in uniq {
                        v.sort_by(|x, z| x.total_cmp(z));
                        a_p.insert(cid, v[v.len() / 2]);
                    }
                    for (obs, obs_mz, cl) in &prows {
                        for &(cid, frag, _) in cl {
                            let mut sub = 0.0f64;
                            for &(pj, _, pij) in cl {
                                if pj != cid {
                                    sub += a_p.get(&pj).copied().unwrap_or(0.0) * pij as f64;
                                }
                            }
                            let cleaned = (*obs as f64 - sub).max(0.0);
                            let e = contested.entry(cid).or_default();
                            e.apportioned += cleaned;
                            if cleaned >= 0.5 * *obs as f64 {
                                e.won += *obs as f64;
                                e.n_won += 1;
                            } else {
                                e.lost += *obs as f64;
                                e.n_lost += 1;
                            }
                            acc2.entry(cid).or_default().push(Hit {
                                rt,
                                frag,
                                inten: cleaned as f32,
                                obs_mz: *obs_mz,
                            });
                        }
                    }
                    continue;
                }
                for peak in &scan.peaks {
                    let inten = peak.intensity;
                    let q_mz = peak.mz / mass_off.factor_at(peak.mz);
                    let obs_mz = peak.mz;
                    claimants.clear();
                    {
                        let mut push = |cid: u32, frag: u16, pi: f32| {
                            let c = cid as usize;
                            if rt < rt_lo[c] || rt > rt_hi[c] {
                                return;
                            }
                            if let Some(s) = restrict {
                                if !s.contains(&cid) {
                                    return;
                                }
                            }
                            claimants.push((cid, frag, pi));
                        };
                        pr.probe(nw.as_mut(), q_mz, lo, hi, &mut push);
                    }
                    if claimants.is_empty() {
                        continue;
                    }
                    let ph = |cid: u32| -> f32 {
                        profile
                            .get(&cid)
                            .and_then(|m| m.get(&rtb))
                            .copied()
                            .unwrap_or(0.0)
                    };
                    // Modular per-claimant competition weight (fragment-competition
                    // framework): the elution profile height, optionally multiplied by
                    // the composable ClaimCues under CoelutionMultiCue. For every other
                    // method the cue is 1.0, so `weights` reduces EXACTLY to `ph` and the
                    // arbitration below is bit-identical to the profile-height version.
                    let multicue = matches!(cfg.peak_claim, PeakClaim::CoelutionMultiCue);
                    let weights: Vec<f32> = claimants
                        .iter()
                        .map(|&(cid, frag, _pi)| {
                            let h = ph(cid);
                            if multicue && h > 0.0 {
                                h * claim_cue_multiplier(
                                    cfg, lib, cid, frag, obs_mz, rt, rt_cal, ms1_scans, ms1_rts,
                                )
                            } else {
                                h
                            }
                        })
                        .collect();
                    let mut best = 0usize;
                    for i in 1..claimants.len() {
                        let (ci, _, pii) = claimants[i];
                        let (cb, _, pib) = claimants[best];
                        let (wi, wb) = (weights[i], weights[best]);
                        if wi > wb || (wi == wb && (pii > pib || (pii == pib && ci < cb))) {
                            best = i;
                        }
                    }
                    let win = claimants[best].0;
                    let sum_w: f32 = weights.iter().copied().sum();
                    let top_w = weights[best];
                    let second_w = claimants
                        .iter()
                        .enumerate()
                        .filter(|&(_, c)| c.0 != win)
                        .map(|(i, _)| weights[i])
                        .fold(0.0f32, f32::max);
                    let dominant =
                        top_w > 0.0 && (second_w <= 0.0 || top_w >= claim_margin * second_w);
                    for (i, &(cid, frag, _pi)) in claimants.iter().enumerate() {
                        let e = contested.entry(cid).or_default();
                        // Weight-proportional share (retained intensity under a
                        // proportional split), tracked for every claimant.
                        let share = if sum_w > 0.0 {
                            inten * (weights[i] / sum_w)
                        } else {
                            inten / claimants.len() as f32
                        };
                        e.apportioned += share as f64;
                        if cid == win {
                            e.won += inten as f64;
                            e.n_won += 1;
                        } else {
                            e.lost += inten as f64;
                            e.n_lost += 1;
                        }
                        if reassign {
                            match cfg.peak_claim {
                                // Destructive MultiCue: winner-take-all on the composite
                                // cue weight (only the best-cue-weighted claimant keeps
                                // the peak). CoelutionWinner is the same rule on the plain
                                // profile height.
                                PeakClaim::CoelutionWinner | PeakClaim::CoelutionMultiCue => {
                                    if cid == win {
                                        acc2.entry(cid).or_default().push(Hit {
                                            rt,
                                            frag,
                                            inten,
                                            obs_mz,
                                        });
                                    }
                                }
                                PeakClaim::CoelutionProportional => {
                                    acc2.entry(cid).or_default().push(Hit {
                                        rt,
                                        frag,
                                        inten: share,
                                        obs_mz,
                                    });
                                }
                                PeakClaim::CoelutionWinnerMargin if !dominant || cid == win => {
                                    acc2.entry(cid).or_default().push(Hit {
                                        rt,
                                        frag,
                                        inten,
                                        obs_mz,
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            let out_acc = if reassign { acc2 } else { acc1 };
            (
                out_acc.into_iter().collect(),
                contested.into_iter().collect(),
            )
        })
        .collect();

    let mut acc: HashMap<u32, Vec<Hit>> = HashMap::new();
    let mut contested: HashMap<u32, Contested> = HashMap::new();
    for (a, c) in partials {
        for (cid, hits) in a {
            acc.entry(cid).or_default().extend(hits);
        }
        for (cid, s) in c {
            let e = contested.entry(cid).or_default();
            e.won += s.won;
            e.lost += s.lost;
            e.n_won += s.n_won;
            e.n_lost += s.n_lost;
            e.apportioned += s.apportioned;
        }
    }
    (acc, contested)
}

pub fn run(p: ExtractParams) -> Result<(u64, u64)> {
    let t0 = Instant::now();
    // Skip the bucketed page_search index when the fragindex backend is selected (the
    // default): it is never read on that path and costs a full sort plus several full
    // copies of every library fragment.
    let build_bucketed = !matches!(p.cfg.matcher, MatcherKind::Fragindex);
    let lib = Library::load_with(
        p.library_precursors,
        p.library_fragments,
        p.cfg.bucket_size,
        build_bucketed,
    )?;

    // Optional candidate allowlist (gate-first-then-compete): restrict extraction to
    // the accepted survivors of a prior gate-on run so the two-pass peak-claim profile
    // map stays small.
    let restrict: Option<std::collections::HashSet<u32>> = match p.restrict_candidates {
        Some(path) => {
            let t = Table::read(path)?;
            let s: std::collections::HashSet<u32> = t.u32("candidate_id")?.into_iter().collect();
            info!(
                restrict_candidates = s.len(),
                "extract: restricting to candidate allowlist"
            );
            Some(s)
        }
        None => None,
    };

    // run windows indexed by candidate_id
    let rw = Table::read(p.run_windows)?;
    let rw_cid = rw.u32("candidate_id")?;
    let rw_cal = rw.f64("rt_pred_cal")?;
    let rw_lo = rw.f64("rt_lo")?;
    let rw_hi = rw.f64("rt_hi")?;
    let ncand = lib.n_candidates();
    let mut rt_lo = vec![f64::NEG_INFINITY; ncand];
    let mut rt_hi = vec![f64::INFINITY; ncand];
    // NaN, not 0.0, for a candidate with no `run_windows` row. Stage B already uses NaN
    // as the explicit "calibration unavailable" sentinel (`candidate_window`), and
    // `calibrated_rt_error` maps a non-finite value to 0.0, i.e. no RT evidence. A 0.0
    // here is *finite*, so it used to produce `rt_error_abs = apex_rt`, the worst possible
    // value, for exactly the candidates the other path gives the best value to. That made
    // the feature a proxy for "was this candidate in the window table", which is not a
    // property of the spectrum. One sentinel, one meaning.
    let mut rt_cal = vec![f64::NAN; ncand];
    for i in 0..rw.nrows {
        let c = rw_cid[i] as usize;
        if c < ncand {
            // The eight RT-window guards downstream are all `rt < rt_lo || rt > rt_hi`,
            // which is false for NaN, so a NaN bound does not reject the scan, it accepts
            // *every* scan: the candidate is searched across the whole gradient with no RT
            // prior and no warning. The legitimate unbounded case is written as explicit
            // -inf/+inf by `candidate_window`, so a NaN here can only mean a corrupt or
            // externally-written window table. Reject it while the row can be named.
            if rw_lo[i].is_nan() || rw_hi[i].is_nan() {
                anyhow::bail!(
                    "run_windows row {i} (candidate_id {c}) has a NaN RT bound \
                     (rt_lo={}, rt_hi={}); an unbounded window must be written as \
                     -inf/+inf, because a NaN bound silently matches every scan instead \
                     of being rejected. Re-run rt-im-train to regenerate {}",
                    rw_lo[i],
                    rw_hi[i],
                    p.run_windows
                );
            }
            rt_lo[c] = rw_lo[i];
            rt_hi[c] = rw_hi[i];
            rt_cal[c] = rw_cal[i];
        }
    }

    let scans = load_ms2(p.ms2)?;
    let ms1_scans: Vec<Ms1Scan> = match p.ms1 {
        Some(path) => load_ms1(path)?,
        None => Vec::new(),
    };
    let ms1_rts: Vec<f64> = ms1_scans.iter().map(|s| s.rt_seconds).collect();
    info!(
        candidates = ncand,
        scans = scans.len(),
        ms1 = ms1_scans.len(),
        "extract: loaded; probing peaks"
    );

    // Isolation-window -> sorted scan RTs, for zero-filled chromatogram grids.
    let windows: Vec<(f64, f64, Vec<f64>)> = if p.cfg.emit_window_grid {
        let mut tmp: HashMap<(u64, u64), Vec<f64>> = HashMap::new();
        for s in &scans {
            tmp.entry((s.window.lower_mz.to_bits(), s.window.upper_mz.to_bits()))
                .or_default()
                .push(s.rt_seconds);
        }
        let mut w: Vec<(f64, f64, Vec<f64>)> = tmp
            .into_iter()
            .map(|((lb, ub), mut v)| {
                v.sort_by(|a, b| a.total_cmp(b));
                (f64::from_bits(lb), f64::from_bits(ub), v)
            })
            .collect();
        w.sort_by(|a, b| a.0.total_cmp(&b.0));
        w
    } else {
        Vec::new()
    };

    // Per-run mass recalibration (optional). Reads the scalar offset + learned
    // tolerance, plus an optional m/z-dependent correction grid (mass_cal_loess).
    let read_grid = |v: &serde_json::Value, key: &str| -> Vec<f64> {
        v.get(key)
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|e| e.as_f64()).collect())
            .unwrap_or_default()
    };
    let (frag_offset, frag_tol, grid_mz, grid_ppm) = match p.mass_cal {
        Some(path) if std::path::Path::new(path).exists() => {
            let v: serde_json::Value = mumdia_io::json::read_json(path)?;
            let off = v
                .get("frag_ppm_offset")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            let tol = v
                .get("frag_tol_ppm")
                .and_then(|x| x.as_f64())
                .unwrap_or(p.cfg.frag_tol_ppm);
            let gmz = read_grid(&v, "mz_cal_grid_mz");
            let gpp = read_grid(&v, "mz_cal_grid_ppm");
            info!(
                frag_ppm_offset = off,
                frag_tol_ppm = tol,
                mz_cal_grid = gmz.len(),
                "extract: using mass recalibration"
            );
            // `extract.frag_tol_ppm` is a FALLBACK, not a setting, in any orchestrated
            // run: search-seed always writes `frag_tol_ppm` into masscal.json --
            // including in its calibration-failure branch, where it writes
            // `search_seed.fragment_tol_ppm` -- and both orchestrators always pass
            // `--mass-cal`. So a config carrying `extract.frag_tol_ppm = 40` extracted at
            // the learned value with nothing said about it. Say it, because a config key
            // that is read and then ignored is worse than one that is absent.
            if (tol - p.cfg.frag_tol_ppm).abs() > 1e-9 {
                warn!(
                    configured_frag_tol_ppm = p.cfg.frag_tol_ppm,
                    learned_frag_tol_ppm = tol,
                    mass_cal = path,
                    "extract: extract.frag_tol_ppm is overridden by the learned tolerance                      from mass calibration. It applies only when no --mass-cal is passed;                      to widen the search tolerance, set search_seed.fragment_tol_ppm"
                );
            }
            (off, tol, gmz, gpp)
        }
        _ => (0.0, p.cfg.frag_tol_ppm, Vec::new(), Vec::new()),
    };
    // The grid is used only if both arrays agree in length and have >= 2 points.
    let mass_off = if grid_mz.len() >= 2 && grid_mz.len() == grid_ppm.len() {
        MassOffset {
            scalar_ppm: frag_offset,
            grid_mz,
            grid_ppm,
        }
    } else {
        MassOffset {
            scalar_ppm: frag_offset,
            grid_mz: Vec::new(),
            grid_ppm: Vec::new(),
        }
    };

    // fragindex backend, built once at the learned fragment tolerance when selected
    // (`MatcherKind::Fragindex`); otherwise the bucketed `Library::page_search` path
    // is used. `Prober::probe` dispatches on this per peak.
    let fidx =
        matches!(p.cfg.matcher, MatcherKind::Fragindex).then(|| FragIndex::build(&lib, frag_tol));

    // Peak-major accumulation.
    let mut acc: HashMap<u32, Vec<Hit>> = HashMap::new();
    // Reused per-peak buffer of (candidate_id, local_frag_index, predicted_intensity).
    let mut claimants: Vec<(u32, u16, f32)> = Vec::new();
    // Per-candidate contested-peak stats under the co-elution arbitration, for the
    // non-destructive soft competition features. Populated only on the two-pass path.
    let mut contested: HashMap<u32, Contested> = HashMap::new();
    // The two co-elution strategies and the contested feature need a first pass to
    // build per-candidate elution profiles before shared peaks can be arbitrated.
    let two_pass = matches!(
        p.cfg.peak_claim,
        PeakClaim::CoelutionWinner
            | PeakClaim::CoelutionProportional
            | PeakClaim::CoelutionWinnerMargin
            | PeakClaim::CoelutionMultiCue
            | PeakClaim::CoelutionDemix
            | PeakClaim::CoelutionShadow
    ) || p.cfg.emit_contested_features;
    let claim_margin = p.cfg.peak_claim_margin as f32;

    if !two_pass {
        if let (Some(idx), true) = (fidx.as_ref(), restrict.is_none()) {
            // Parallel across isolation-window groups (bit-identical to serial: the
            // cascade rt-sorts each candidate's hits before summing). Only when there
            // is no candidate allowlist; a `restrict` list routes to the serial path
            // below, which applies the allowlist filter and honors every peak_claim
            // strategy (Winner/Proportional/None).
            acc = extract_accumulate_windows(idx, &scans, &rt_lo, &rt_hi, &mass_off, p.cfg);
        } else {
            let pr = Prober {
                fidx: fidx.as_ref(),
                lib: &lib,
                frag_tol,
            };
            for scan in &scans {
                let (lo, hi) = lib.candidate_range(scan.window.lower_mz, scan.window.upper_mz);
                if hi <= lo {
                    continue;
                }
                let rt = scan.rt_seconds;
                for peak in &scan.peaks {
                    let inten = peak.intensity;
                    let q_mz = peak.mz / mass_off.factor_at(peak.mz);
                    // Collect every co-isolated, in-RT-window candidate matching this
                    // peak, then apportion per the claim strategy. In wide DIA one peak
                    // matches many candidates (~98% of fragments collide).
                    claimants.clear();
                    {
                        let mut push = |cid: u32, frag: u16, pi: f32| {
                            let c = cid as usize;
                            if rt < rt_lo[c] || rt > rt_hi[c] {
                                return;
                            }
                            if let Some(s) = &restrict {
                                if !s.contains(&cid) {
                                    return;
                                }
                            }
                            claimants.push((cid, frag, pi));
                        };
                        pr.probe(None, q_mz, lo, hi, &mut push);
                    }
                    if claimants.is_empty() {
                        continue;
                    }
                    let obs_mz = peak.mz;
                    match p.cfg.peak_claim {
                        PeakClaim::WinnerPredictedIntensity => {
                            let mut best = 0usize;
                            for i in 1..claimants.len() {
                                let a = claimants[i];
                                let b = claimants[best];
                                if a.2 > b.2 || (a.2 == b.2 && a.0 < b.0) {
                                    best = i;
                                }
                            }
                            let (cid, frag, _) = claimants[best];
                            acc.entry(cid).or_default().push(Hit {
                                rt,
                                frag,
                                inten,
                                obs_mz,
                            });
                        }
                        PeakClaim::Proportional => {
                            let sump: f32 = claimants.iter().map(|c| c.2.max(0.0)).sum();
                            for &(cid, frag, pi) in &claimants {
                                let share = if sump > 0.0 {
                                    inten * (pi.max(0.0) / sump)
                                } else {
                                    inten / claimants.len() as f32
                                };
                                acc.entry(cid).or_default().push(Hit {
                                    rt,
                                    frag,
                                    inten: share,
                                    obs_mz,
                                });
                            }
                        }
                        // None (and the co-elution variants, which never reach here).
                        _ => {
                            for &(cid, frag, _) in &claimants {
                                acc.entry(cid).or_default().push(Hit {
                                    rt,
                                    frag,
                                    inten,
                                    obs_mz,
                                });
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Two-pass co-elution peak-claim, parallelized across isolation windows
        // (each window's candidates interact only within it, so the two expensive
        // probing passes fan out over the ~150 windows).
        // CoelutionMultiCue ships non-destructive by default (cue-weighted split flows
        // into the contested/apportioned FEATURES only). It becomes destructive, rewriting
        // the extracted intensities so ALL downstream features recompute on the competed
        // evidence, only when `claim_cues.reassign` is set (entrapment-gated).
        let reassign = matches!(
            p.cfg.peak_claim,
            PeakClaim::CoelutionWinner
                | PeakClaim::CoelutionProportional
                | PeakClaim::CoelutionWinnerMargin
                | PeakClaim::CoelutionDemix
                | PeakClaim::CoelutionShadow
        ) || (matches!(p.cfg.peak_claim, PeakClaim::CoelutionMultiCue)
            && p.cfg.claim_cues.reassign);
        let (a, c) = extract_twopass_windows(
            fidx.as_ref(),
            &lib,
            &scans,
            &rt_lo,
            &rt_hi,
            &rt_cal,
            &ms1_scans,
            &ms1_rts,
            &mass_off,
            frag_tol,
            p.cfg,
            restrict.as_ref(),
            reassign,
            claim_margin,
        );
        acc = a;
        contested = c;
    }
    info!(
        materialized = acc.len(),
        "extract: candidates with evidence"
    );

    // Cascade + apex per candidate.
    let scan_window = p.cfg.fixed_scan_window.max(1);

    // psms_extracted columns
    let (mut cid_c, mut apexrt_c, mut apexint_c) = (Vec::new(), Vec::new(), Vec::new());
    // Top-K peak promotion (AlphaDIA #7): the peak rank of each emitted PSM row. 0 is
    // the selected apex (the only row per candidate until promote_top_peaks > 1).
    let mut peakrank_c: Vec<i32> = Vec::new();
    let (mut nmatch_c, mut corun_c, mut npred_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut calrt_c, mut mz_c, mut z_c) = (Vec::new(), Vec::new(), Vec::new());
    let (mut label_c, mut base_c, mut pform_c, mut prot_c, mut irt_c) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut apexim_c: Vec<Option<f64>> = Vec::new();
    // Fraction of this candidate's matched intensity that a co-eluting competitor
    // claims more strongly (co-elution arbitration); 0 when the two-pass path is off.
    let mut contested_c: Vec<f64> = Vec::new();
    // Richer soft-competition columns, emitted only with emit_contested_features.
    let (mut contested_count_c, mut apportioned_c): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    // MS1 apex isotope intensities (null when no MS1 provided).
    let mut ms1_m1: Vec<Option<f64>> = Vec::new();
    let mut ms1_mono: Vec<Option<f64>> = Vec::new();
    let mut ms1_i1: Vec<Option<f64>> = Vec::new();
    let mut ms1_i2: Vec<Option<f64>> = Vec::new();
    // Gate diagnostic scores (per accepted candidate; see CandOut).
    let (mut gate_apex_c, mut gate_peakspec_c, mut gate_coel_c, mut gate_se_c): (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // Demix (D1/D2/D3) feature columns.
    #[allow(clippy::type_complexity)]
    let (
        mut deconv_expl_c,
        mut deconv_act_c,
        mut deconv_share_c,
        mut deconv_collin_c,
        mut deconv_shadow_c,
    ): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

    // chromatograms columns
    let (mut ch_cid, mut ch_name, mut ch_fmz, mut ch_pint) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut ch_obsmz, mut ch_rt, mut ch_int): (Vec<f64>, Vec<Vec<f32>>, Vec<Vec<f32>>) =
        (Vec::new(), Vec::new(), Vec::new());

    // Deterministic output order (a HashMap's iteration order is randomized,
    // and downstream floating-point sums in the rescorer are order-sensitive).
    let mut cand_ids: Vec<u32> = acc.keys().cloned().collect();
    cand_ids.sort_unstable();

    // Drain the accumulator into (cid, hits) in the deterministic sorted order,
    // then process candidates in parallel. Each candidate's work depends only on
    // its own hits plus read-only library/window/MS1 data, and every float
    // reduction (apex sum, wsum, chromatogram grids) is self-contained, so
    // collecting the results in cand_ids order and appending them serially below
    // yields output (PSM rows and chromatogram rows) byte-identical to the serial
    // loop. The optional Pearson gate now allocates its two scratch vectors per
    // candidate (was a hoisted reused buffer) because buffers cannot be shared
    // across parallel candidates.
    let cand_hits: Vec<(u32, Vec<Hit>)> = cand_ids
        .iter()
        .map(|&cid| (cid, acc.remove(&cid).unwrap()))
        .collect();

    struct CandOut {
        cid: u32,
        /// Chromatographic peak rank (0 = selected apex). Top-K promotion (#7).
        peak_rank: u8,
        apex_rt: f64,
        apex_int: f32,
        n_match: i32,
        corun: i32,
        npred: i32,
        calrt: f64,
        mz: f64,
        contested: f64,
        contested_count_frac: f64,
        apportioned_frac: f64,
        z: i32,
        label: String,
        base: u32,
        pform: String,
        prot: String,
        irt: f32,
        ms1_m1: Option<f64>,
        ms1_mono: Option<f64>,
        ms1_i1: Option<f64>,
        ms1_i2: Option<f64>,
        /// Gate diagnostic scores, computed for EVERY accepted candidate regardless
        /// of `gate_mode` (sensitivity program): the single-apex-scan intensity
        /// Pearson, the peak-integrated spectral Pearson, and the temporal co-elution
        /// score. Emitted so an offline analysis can compare gate metrics (and their
        /// combination) at matched pool size, without re-extraction.
        gate_apex: f32,
        gate_peak_spectral: f32,
        gate_coelution: f32,
        gate_spectral_entropy: f32,
        /// Spectrum-centric demix features (D2), all 0 unless `emit_demix_features`:
        /// residual-explained fraction, active-set survival flag, and this candidate's
        /// fraction of the total demixed abundance at its apex.
        deconv_explained: f32,
        deconv_active: f32,
        deconv_share: f32,
        deconv_collin: f32,
        deconv_shadow: f32,
        /// (cid, frag_name, frag_mz, frag_obs_mz, predicted_intensity, rt, intensity)
        chrom: Vec<ChromOutputRow>,
        /// Top-K retained peak groups (sensitivity_plan P1.1/P1.2), populated only
        /// when `retain_top_peaks > 1`. Each: (rank, apex_rt, start_rt, end_rt,
        /// evidence_count, area). Ranked by co-eluting fragment breadth (not
        /// intensity). The main PSM above still reports the single selected apex,
        /// so FDR is unaffected; these are candidate peaks for an offline peak-
        /// selection model. Empty for K=1.
        peaks: Vec<(u8, f64, f64, f64, f64, f64)>,
    }

    // Apex-scan lookup for spectrum-centric demixing (D2): rt_bits -> scan indices.
    // Built only when demixing is requested, so the default path pays nothing.
    let rt_scan: HashMap<u64, Vec<u32>> = if p.cfg.emit_demix_features {
        let mut m: HashMap<u64, Vec<u32>> = HashMap::new();
        for (si, s) in scans.iter().enumerate() {
            m.entry(s.rt_seconds.to_bits()).or_default().push(si as u32);
        }
        m
    } else {
        HashMap::new()
    };

    let results: Vec<Vec<CandOut>> = cand_hits
        .into_par_iter()
        .map(|(cid, mut hits)| {
            // distinct matched fragments (tier b)
            let mut distinct: Vec<u16> = hits.iter().map(|h| h.frag).collect();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.len() < p.cfg.presence_min_matched.max(1) {
                return Vec::new();
            }

            // Group hits into scan groups by RT (dedupe same fragment in a scan by max).
            hits.sort_by(|a, b| a.rt.total_cmp(&b.rt));
            // scan groups: Vec<(rt, BTreeMap<frag,intensity>)>. A BTreeMap keeps the
            // per-scan fragment order fixed so the f32 apex sum is deterministic.
            let mut groups: Vec<(f64, BTreeMap<u16, f32>)> = Vec::new();
            for h in &hits {
                match groups.last_mut() {
                    Some((rt, map)) if (*rt - h.rt).abs() < 1e-9 => {
                        let e = map.entry(h.frag).or_insert(0.0);
                        if h.inten > *e {
                            *e = h.inten;
                        }
                    }
                    _ => {
                        let mut m = BTreeMap::new();
                        m.insert(h.frag, h.inten);
                        groups.push((h.rt, m));
                    }
                }
            }

            let (fmzs0, fints0, _) = lib.cand_frags(cid);

            // Acquisition scan grid: the covering isolation-window scans within the
            // RT window. Project the sparse hit-groups onto it so apex counting and
            // the co-elution run see MISSING acquisition scans (count 0, and they
            // break a run) rather than only scans that happened to carry a hit. When
            // no covering-window grid is available, fall back to the sparse groups.
            let grid: Vec<f64> = if !windows.is_empty() {
                let pm = lib.cands[cid as usize].precursor_mz;
                let (lo, hi) = (rt_lo[cid as usize], rt_hi[cid as usize]);
                let mut g: Vec<f64> = Vec::new();
                for (wl, wu, rts) in &windows {
                    if *wl <= pm && pm <= *wu {
                        let a = rts.partition_point(|&r| r < lo);
                        let b = rts.partition_point(|&r| r <= hi);
                        g.extend_from_slice(&rts[a..b]);
                    }
                }
                g.sort_by(|a, b| a.total_cmp(b));
                g.dedup();
                g
            } else {
                Vec::new()
            };
            if !grid.is_empty() {
                let g2i: HashMap<u64, usize> = grid
                    .iter()
                    .enumerate()
                    .map(|(j, r)| (r.to_bits(), j))
                    .collect();
                let mut aligned: Vec<(f64, BTreeMap<u16, f32>)> =
                    grid.iter().map(|&r| (r, BTreeMap::new())).collect();
                for (rt, map) in std::mem::take(&mut groups) {
                    if let Some(&j) = g2i.get(&rt.to_bits()) {
                        aligned[j].1 = map;
                    }
                }
                groups = aligned;
            }

            // Apex: the scan group with the most distinct matched fragments, allowing
            // scans within `apex_count_tol` of that maximum (so a slightly-lower-count
            // but much more intense scan can still win), then the one maximizing the
            // summed intensity of its 3 most intense fragments. This is the diagnostic-
            // plot apex; robust to a bright single-fragment interferent that would win a
            // pure max-summed-intensity apex in chimeric DIA.
            // Distinct-fragment count per scan group, optionally smoothed by a centered
            // rolling SUM (`apex_count_window`). Low-intensity fragments flicker in/out
            // scan-to-scan; a single-scan count then spikes at noise scans and misplaces
            // the apex. The rolling sum makes the apex land in the region of *sustained*
            // fragment presence. It is deliberately a sum, not a mean: the window is
            // truncated at the profile edges, so interior positions accumulate more than
            // edge positions, which center-weights the apex toward the RT-window centre
            // (~= the predicted RT). That mild RT-prior steers off off-centre interfering
            // peaks; measured, sum beats mean by ~+300 IDs on the AIF file. Window 1
            // reproduces the exact per-scan-count behavior.
            let counts: Vec<usize> = groups.iter().map(|(_, m)| m.len()).collect();
            let w = p.cfg.apex_count_window.max(1);
            let r = w / 2;
            let sigma = p.cfg.apex_gaussian_sigma_scans;
            // Smoothed per-scan fragment-count score. Default is the truncated
            // rolling SUM (`apex_count_window`); with `apex_gaussian_sigma_scans` > 0
            // a Gaussian matched filter (radius 3*sigma) is used instead. Both are
            // deterministic and reduce to the raw per-scan count when disabled.
            let smoothed: Vec<f64> = if sigma > 0.0 {
                let radius = (sigma * 3.0).ceil() as usize;
                let kernel: Vec<f64> = (0..=2 * radius)
                    .map(|k| {
                        let d = k as f64 - radius as f64;
                        (-0.5 * (d / sigma).powi(2)).exp()
                    })
                    .collect();
                (0..counts.len())
                    .map(|i| {
                        let mut acc = 0.0;
                        for (k, &wt) in kernel.iter().enumerate() {
                            let idx = i as isize + k as isize - radius as isize;
                            if idx >= 0 && (idx as usize) < counts.len() {
                                acc += counts[idx as usize] as f64 * wt;
                            }
                        }
                        acc
                    })
                    .collect()
            } else if w <= 1 {
                counts.iter().map(|&c| c as f64).collect()
            } else {
                (0..counts.len())
                    .map(|i| {
                        let lo = i.saturating_sub(r);
                        let hi = (i + r).min(counts.len() - 1);
                        counts[lo..=hi].iter().sum::<usize>() as f64
                    })
                    .collect()
            };
            let maxc = smoothed.iter().copied().fold(0.0f64, f64::max);
            let thresh = (maxc - p.cfg.apex_count_tol as f64).max(0.0);
            // Optional Gaussian RT prior on the apex tiebreak: among count-qualified
            // scans, multiply the top-3 intensity by exp(-0.5*((rt - rt_pred_cal)/sigma)^2)
            // so a distant-from-prediction interferent inside a wide RT window cannot
            // define the apex. sigma = `apex_rt_prior_s`; 0 (or an unset rt_cal) disables it.
            let rt_prior_sigma = p.cfg.apex_rt_prior_s;
            let rt_cal_c = rt_cal[cid as usize];
            let use_prior = rt_prior_sigma > 0.0 && rt_cal_c > 0.0;
            // Signature-ion apex tiebreak: sum the OBSERVED intensity of the top-K
            // PREDICTED fragments (`apex_top_fragments`; 0 -> a default of 3) at each
            // qualifying scan, instead of the 3 brightest observed peaks. A bright
            // interferent on a non-signature ion can then no longer define the apex.
            let k_sig = if p.cfg.apex_top_fragments > 0 {
                p.cfg.apex_top_fragments
            } else {
                3
            };
            let sig: Vec<u16> = {
                let mut ord: Vec<usize> = (0..fints0.len()).collect();
                ord.sort_by(|&a, &b| fints0[b].total_cmp(&fints0[a]));
                ord.into_iter().take(k_sig).map(|o| o as u16).collect()
            };
            let mut apex_rt = groups[0].0;
            let mut apex_sum = 0.0f32;
            let mut best_sig = f32::NEG_INFINITY;
            for (i, (rt, map)) in groups.iter().enumerate() {
                if map.is_empty() || smoothed[i] < thresh {
                    continue;
                }
                let sig_sum: f32 = sig
                    .iter()
                    .map(|&o| map.get(&o).copied().unwrap_or(0.0))
                    .sum();
                let prior = if use_prior {
                    (-0.5 * ((*rt - rt_cal_c) / rt_prior_sigma).powi(2)).exp() as f32
                } else {
                    1.0
                };
                let score = if p.cfg.apex_evidence_rank {
                    // Breadth-of-evidence apex: the count of distinct co-eluting
                    // predicted fragments at this scan dominates; observed signature
                    // intensity only breaks ties within [0,1). Interference-resistant
                    // in wide-window DIA (a chimeric-intensity spike cannot outvote a
                    // scan where more of the peptide's own transitions co-elute).
                    let n_frag = map.len() as f32;
                    let tie = sig_sum / (sig_sum + 1.0);
                    (n_frag + tie) * prior
                } else {
                    // Legacy: signature-ion observed intensity (x RT prior). Bit-identical
                    // to the previous behaviour (prior = 1.0 when the RT prior is off).
                    sig_sum * prior
                };
                if score > best_sig {
                    best_sig = score;
                    apex_rt = *rt;
                    apex_sum = map.values().sum(); // report full apex intensity
                }
            }

            // Co-elution run: max consecutive scan groups with >= min_coelution frags.
            let mut best_run = 0usize;
            let mut cur = 0usize;
            for (_, map) in &groups {
                if map.len() >= p.cfg.presence_min_coelution.max(1) {
                    cur += 1;
                    best_run = best_run.max(cur);
                } else {
                    cur = 0;
                }
            }

            // Acceptance (tier c): presence, consecutive-scan run, and matched
            // fraction of the predicted fragments (symmetric discriminator).
            let matched_fraction = distinct.len() as f64 / (fmzs0.len().max(1) as f64);
            if distinct.len() < p.cfg.presence_min_fragments.max(1)
                || best_run < scan_window
                || best_run < p.cfg.min_coelution_run
                || matched_fraction < p.cfg.min_matched_fraction
            {
                return Vec::new();
            }

            let c = &lib.cands[cid as usize];

            // MS1 apex isotope intensities at a given RT (nearest MS1 scan). Factored
            // so both the selected apex (rank 0) and any promoted alternate peak (#7)
            // compute their own MS1 evidence at their own apex RT. Computed BEFORE the
            // acceptance gate so MS1 evidence can rescue a candidate the single-scan
            // fragment-Pearson gate would otherwise reject.
            let ms1_at = |rt: f64| -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
                if ms1_scans.is_empty() {
                    return (None, None, None, None);
                }
                let j = nearest_index(&ms1_rts, rt);
                let s = &ms1_scans[j];
                let z = c.charge as f64;
                let sp = ISOTOPE_SPACING / z;
                let tol = p.cfg.prec_tol_ppm;
                (
                    Some(sum_near(&s.mz, &s.intensity, c.precursor_mz - sp, tol) as f64),
                    Some(sum_near(&s.mz, &s.intensity, c.precursor_mz, tol) as f64),
                    Some(sum_near(&s.mz, &s.intensity, c.precursor_mz + sp, tol) as f64),
                    Some(sum_near(&s.mz, &s.intensity, c.precursor_mz + 2.0 * sp, tol) as f64),
                )
            };
            let (o_ms1_m1, o_ms1_mono, o_ms1_i1, o_ms1_i2) = ms1_at(apex_rt);
            // Cheap MS1 support: mono present and the +1/mono ratio in a plausible
            // averagine band. Used only as the rescue signal for the Pearson gate.
            let ms1_support = {
                let mono = o_ms1_mono.unwrap_or(0.0);
                let i1 = o_ms1_i1.unwrap_or(0.0);
                mono > 0.0 && i1 > 0.0 && {
                    let r = i1 / mono;
                    (0.1..=1.5).contains(&r)
                }
            };

            // Optional tier-d Pearson gate (kept for configurability; matched fraction
            // above is the primary symmetric discriminator). With `ms1_rescue`, a
            // candidate that fails the single-scan fragment Pearson is kept when it has
            // adequate matched fragments AND MS1 isotope-pattern support.
            let apex_map = groups
                .iter()
                .find(|(rt, _)| (*rt - apex_rt).abs() < 1e-9)
                .map(|(_, m)| m);
            // Spectral-agreement score closures, evaluated lazily: the acceptance gate
            // needs only the ACTIVE `gate_mode`'s score, and the four diagnostic scores
            // are computed only when `emit_gate_diagnostics` is set (see below), so the
            // default chain pays the same per-candidate cost as before this feature.
            let apex_obs: Option<Vec<f64>> = apex_map.map(|map| {
                (0..fmzs0.len())
                    .map(|k| *map.get(&(k as u16)).unwrap_or(&0.0) as f64)
                    .collect()
            });
            let pred_f64: Vec<f64> = fints0.iter().map(|x| *x as f64).collect();
            // Single-apex-scan intensity Pearson (1.0 when no apex scan resolved -> do
            // not reject on spectral agreement).
            let apex_pearson = || match &apex_obs {
                Some(obs) => crate::stats::pearson(obs, &pred_f64),
                None => 1.0,
            };
            // spectral_entropy_similarity_sqrt of the apex spectrum (shared kernel in
            // features::entropy; best single target/decoy gate discriminator).
            let apex_entropy = || match &apex_obs {
                Some(obs) => crate::stages::features::entropy::spectral_entropy_similarity_sqrt(
                    obs, &pred_f64,
                ),
                None => 1.0,
            };
            let peak_spec = || peak_spectral_score(&groups, &sig, fints0);
            let coel = || coelution_gate_score(&groups, &distinct, &sig, fints0);

            if p.cfg.gate_min_score > 0.0 {
                // Acceptance gate. `gate_min_score` thresholds the ACTIVE gate_mode's
                // spectral-agreement score: the legacy single-apex-scan Pearson (one
                // chimeric scan can dominate), the peak-integrated spectral Pearson, the
                // apex spectral-entropy similarity, the temporal co-elution score, or
                // Combined (both, more specific). Only the active score computes.
                let rejected = match p.cfg.gate_mode {
                    GateMode::ApexPearson => apex_pearson() < p.cfg.gate_min_score,
                    GateMode::PeakSpectral => peak_spec() < p.cfg.gate_min_score,
                    GateMode::SpectralEntropy => apex_entropy() < p.cfg.gate_min_score,
                    GateMode::Coelution => coel() < p.cfg.gate_min_score,
                    GateMode::Combined => {
                        peak_spec() < p.cfg.gate_min_score || coel() < p.cfg.gate_coelution_min
                    }
                };
                if rejected {
                    let rescued = p.cfg.ms1_rescue
                        && ms1_support
                        && distinct.len() >= p.cfg.presence_min_fragments.max(1);
                    if !rescued {
                        return Vec::new();
                    }
                }
            }

            // Diagnostic scores (all four metrics, for the offline gate-metric
            // comparison). Computed and emitted ONLY when `emit_gate_diagnostics` is set,
            // so the default psms.parquet schema and per-candidate compute are unchanged
            // (sensitivity-program: default-off, byte-identical). Zero when off (the four
            // columns are not written).
            let (gate_apex, gate_peak_spectral, gate_coelution, gate_spectral_entropy) =
                if p.cfg.emit_gate_diagnostics {
                    (apex_pearson(), peak_spec(), coel(), apex_entropy())
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

            // Soft competition features from the co-elution arbitration (all 0 when the
            // two-pass path did not run). contested_frac: fraction of contested INTENSITY
            // lost to better co-eluters. contested_count_frac: fraction of contested
            // fragment-PEAKS lost. apportioned_frac: fraction of contested intensity the
            // candidate retains under proportional apportionment (1 = keeps all, ~0 = a
            // peak-borrower stripped by its co-eluting competitors).
            let cst = contested.get(&cid).copied().unwrap_or_default();
            let contested_val = {
                let t = cst.won + cst.lost;
                if t > 0.0 {
                    cst.lost / t
                } else {
                    0.0
                }
            };
            let contested_count_frac = {
                let n = cst.n_won + cst.n_lost;
                if n > 0 {
                    cst.n_lost as f64 / n as f64
                } else {
                    0.0
                }
            };
            let apportioned_frac = {
                let t = cst.won + cst.lost;
                if t > 0.0 {
                    cst.apportioned / t
                } else {
                    0.0
                }
            };

            // Per-fragment intensity-weighted observed m/z (for mass accuracy).
            let mut wsum: HashMap<u16, (f64, f64)> = HashMap::new(); // frag -> (sum w*mz, sum w)
            for h in &hits {
                let e = wsum.entry(h.frag).or_insert((0.0, 0.0));
                e.0 += h.obs_mz * h.inten as f64;
                e.1 += h.inten as f64;
            }

            let mut chrom_rows: Vec<ChromOutputRow> = Vec::new();

            // Emit chromatograms. When emit_window_grid is set, each fragment is sampled
            // on the full isolation-window scan grid (all scans of the covering window(s)
            // within the RT window), with 0.0 where the fragment is absent, so the elution
            // profile drops to zero between peaks (correct boundary calling downstream).
            let (fmzs, fints, fnames) = lib.cand_frags(cid);
            let mut per_frag: HashMap<u16, Vec<(f64, f32)>> = HashMap::new();
            for (rt, map) in &groups {
                for (&frag, &inten) in map {
                    per_frag.entry(frag).or_default().push((*rt, inten));
                }
            }
            // (the acquisition-scan `grid` was computed above, before apex/co-elution)
            // Emit a row for EVERY predicted transition so the feature families see the
            // full predicted set (a missing strong ion is penalized). An OBSERVED
            // fragment carries its grid-sampled (or sorted) trace; a NEVER-OBSERVED one
            // carries an EMPTY trace, NOT a grid-length zero vector. The empty trace
            // still yields obs_apex = 0 downstream, and keeps the total chromatogram
            // list-value count down (a grid-length zero per absent fragment would
            // bloat it needlessly; the column itself is now a 64-bit LargeList, so the
            // old ~2.1B 32-bit offset ceiling no longer applies).
            // obs m/z falls back to theoretical; harmless since mass-accuracy counts
            // only fragments with obs_apex > 0.
            for fi in 0..fmzs.len() {
                let frag = fi as u16;
                let obs_mz = wsum
                    .get(&frag)
                    .map(|(sm, sw)| if *sw > 0.0 { sm / sw } else { fmzs[fi] })
                    .unwrap_or(fmzs[fi]);
                let (rts, ints): (Vec<f32>, Vec<f32>) = match per_frag.get(&frag) {
                    Some(v) if !grid.is_empty() => {
                        let m: HashMap<u64, f32> =
                            v.iter().map(|(r, i)| (r.to_bits(), *i)).collect();
                        (
                            grid.iter().map(|r| *r as f32).collect(),
                            grid.iter()
                                .map(|r| *m.get(&r.to_bits()).unwrap_or(&0.0))
                                .collect(),
                        )
                    }
                    Some(v) => {
                        let mut s = v.clone();
                        s.sort_by(|a, b| a.0.total_cmp(&b.0));
                        (
                            s.iter().map(|(r, _)| *r as f32).collect(),
                            s.iter().map(|(_, i)| *i).collect(),
                        )
                    }
                    None => (Vec::new(), Vec::new()), // absent predicted transition
                };
                chrom_rows.push((
                    cid,
                    // Fragment names are interned in the library (a u16 dictionary id per
                    // fragment instead of a String); materialise the String only here, for
                    // the emitted chromatogram row.
                    lib.frag_name_str(fnames[fi]).to_string(),
                    fmzs[fi],
                    obs_mz,
                    fints[fi],
                    rts,
                    ints,
                ));
            }

            // MS1 isotope XICs (mono/+1/+2) sampled on the same scan grid as the
            // fragments, so the features stage can correlate the MS1 precursor
            // envelope against the MS2 fragments over the elution peak (DIA-NN
            // Ms1.Profile.Corr class). Grid mode only; nearest MS1 scan per grid RT.
            if !ms1_scans.is_empty() && !grid.is_empty() {
                let sp = ISOTOPE_SPACING / c.charge as f64;
                let tol = p.cfg.prec_tol_ppm;
                let grid_rt: Vec<f32> = grid.iter().map(|&r| r as f32).collect();
                for (nm, dmz) in [("ms1_mono", 0.0), ("ms1_iso1", sp), ("ms1_iso2", 2.0 * sp)] {
                    let mz = c.precursor_mz + dmz;
                    let ints: Vec<f32> = grid
                        .iter()
                        .map(|&r| {
                            let j = nearest_index(&ms1_rts, r);
                            sum_near(&ms1_scans[j].mz, &ms1_scans[j].intensity, mz, tol) as f32
                        })
                        .collect();
                    chrom_rows.push((cid, nm.to_string(), mz, mz, 0.0, grid_rt.clone(), ints));
                }
            }

            // Top-K peak retention (opt-in; sensitivity_plan P1.1/P1.2). Enumerate peak
            // groups over the per-scan distinct-fragment COUNT profile (co-eluting
            // breadth, interference-resistant per the intensity-is-chimeric argument),
            // ranked by breadth-area. The PSM above still carries the selected apex, so
            // FDR is unchanged; these are extra candidate peaks for an offline peak-
            // selection model. Empty for K=1 (the default).
            let peaks: Vec<(u8, f64, f64, f64, f64, f64)> = if p.cfg.retain_top_peaks > 1
                && !groups.is_empty()
            {
                let count_prof: Vec<f32> = groups.iter().map(|(_, m)| m.len() as f32).collect();
                crate::peaks::enumerate_peaks(&count_prof, p.cfg.retain_top_peaks, 1.0 / 3.0, 0.1)
                    .into_iter()
                    .map(|pk| {
                        (
                            pk.rank as u8,
                            groups[pk.apex_idx].0,
                            groups[pk.start_idx].0,
                            groups[pk.end_idx].0,
                            groups[pk.apex_idx].1.len() as f64,
                            pk.area as f64,
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // Spectrum-centric NNLS demixing at the selected apex (D2). Non-destructive:
            // emits interference-corrected features only. Gated; zero when off.
            //
            // Filled in a second pass below, not here: the whole problem is a function of
            // the apex SCAN, so solving it inside this per-candidate loop re-probed every
            // peak of that scan and re-ran the NNLS once per candidate sharing it.
            let (deconv_explained, deconv_active, deconv_share, deconv_collin, deconv_shadow) =
                (0.0, 0.0, 0.0, 0.0, 0.0);

            let rank0 = CandOut {
                cid,
                peak_rank: 0, // selected apex; ranks >= 1 added when promote_top_peaks > 1
                apex_rt,
                apex_int: apex_sum,
                n_match: distinct.len() as i32,
                corun: best_run as i32,
                npred: fmzs0.len() as i32,
                calrt: rt_cal[cid as usize],
                mz: c.precursor_mz,
                contested: contested_val,
                contested_count_frac,
                apportioned_frac,
                z: c.charge,
                label: if c.is_decoy { "decoy" } else { "target" }.to_string(),
                base: c.base_peptide_id,
                pform: c.peptidoform.clone(),
                prot: c.protein.clone(),
                irt: c.predicted_irt,
                ms1_m1: o_ms1_m1,
                ms1_mono: o_ms1_mono,
                ms1_i1: o_ms1_i1,
                ms1_i2: o_ms1_i2,
                gate_apex: gate_apex as f32,
                gate_peak_spectral: gate_peak_spectral as f32,
                gate_coelution: gate_coelution as f32,
                gate_spectral_entropy: gate_spectral_entropy as f32,
                deconv_explained: deconv_explained as f32,
                deconv_active: deconv_active as f32,
                deconv_share: deconv_share as f32,
                deconv_collin: deconv_collin as f32,
                deconv_shadow: deconv_shadow as f32,
                chrom: chrom_rows,
                peaks,
            };
            if p.cfg.promote_top_peaks <= 1 || groups.is_empty() {
                return vec![rank0];
            }
            // Promote alternate chromatographic peaks (#7). Enumerate on the same
            // distinct-fragment COUNT profile as the diagnostic peaks (breadth of
            // co-elution, interference-resistant), exclude the envelope holding the
            // selected apex, gate each candidate peak by area fraction + apex-RT
            // separation + matched-fragment floor, and emit up to promote_top_peaks - 1
            // extra records. Each shares the candidate's chromatograms (emitted once on
            // rank 0, looked up by candidate_id downstream) and re-slices only its own
            // apex-dependent scalars + MS1; the features stage recomputes peak-shape and
            // co-elution features from the shared chrom windowed to each row's own apex.
            let count_prof: Vec<f32> = groups.iter().map(|(_, m)| m.len() as f32).collect();
            let alt_peaks =
                crate::peaks::enumerate_peaks(&count_prof, p.cfg.promote_top_peaks, 1.0 / 3.0, 0.1);
            let apex_gi = groups
                .iter()
                .position(|(rt, _)| (*rt - apex_rt).abs() < 1e-9);
            // Reference area for the area gate: the enumerated envelope holding the
            // selected apex (0 disables the area gate if the apex is not a counted peak).
            let rank0_area = apex_gi
                .and_then(|gi| {
                    alt_peaks
                        .iter()
                        .find(|pk| pk.start_idx <= gi && gi <= pk.end_idx)
                })
                .map(|pk| pk.area as f64)
                .unwrap_or(0.0);
            let mut out = vec![rank0];
            let mut rank: u8 = 1;
            for pk in &alt_peaks {
                if out.len() >= p.cfg.promote_top_peaks {
                    break;
                }
                // Exclude the envelope containing the selected apex (rank 0).
                if let Some(gi) = apex_gi {
                    if pk.start_idx <= gi && gi <= pk.end_idx {
                        continue;
                    }
                }
                let alt_apex_rt = groups[pk.apex_idx].0;
                if (alt_apex_rt - apex_rt).abs() < p.cfg.alt_peak_min_separation_s {
                    continue;
                }
                if rank0_area > 0.0 && (pk.area as f64) < p.cfg.alt_peak_min_area_frac * rank0_area
                {
                    continue;
                }
                let mut altset: std::collections::HashSet<u16> = std::collections::HashSet::new();
                for (_, m) in &groups[pk.start_idx..=pk.end_idx] {
                    for &f in m.keys() {
                        altset.insert(f);
                    }
                }
                if altset.len() < p.cfg.presence_min_matched.max(1) {
                    continue;
                }
                let alt_apex_int: f32 = groups[pk.apex_idx].1.values().sum();
                let (a_m1, a_mono, a_i1, a_i2) = ms1_at(alt_apex_rt);
                out.push(CandOut {
                    cid,
                    peak_rank: rank,
                    apex_rt: alt_apex_rt,
                    apex_int: alt_apex_int,
                    n_match: altset.len() as i32,
                    corun: best_run as i32,
                    npred: fmzs0.len() as i32,
                    calrt: rt_cal[cid as usize],
                    mz: c.precursor_mz,
                    contested: contested_val,
                    contested_count_frac,
                    apportioned_frac,
                    z: c.charge,
                    label: if c.is_decoy { "decoy" } else { "target" }.to_string(),
                    base: c.base_peptide_id,
                    pform: c.peptidoform.clone(),
                    prot: c.protein.clone(),
                    irt: c.predicted_irt,
                    ms1_m1: a_m1,
                    ms1_mono: a_mono,
                    ms1_i1: a_i1,
                    ms1_i2: a_i2,
                    gate_apex: gate_apex as f32,
                    gate_peak_spectral: gate_peak_spectral as f32,
                    gate_coelution: gate_coelution as f32,
                    gate_spectral_entropy: gate_spectral_entropy as f32,
                    // Demix is a rank-0 apex feature; alternate peaks carry 0.
                    deconv_explained: 0.0,
                    deconv_active: 0.0,
                    deconv_share: 0.0,
                    deconv_collin: 0.0,
                    deconv_shadow: 0.0,
                    chrom: Vec::new(), // shared per-candidate via the rank-0 row
                    peaks: Vec::new(),
                });
                rank += 1;
            }
            out
        })
        .collect();

    // Spectrum-centric NNLS demixing (D2), second pass: solve ONCE PER APEX SCAN.
    //
    // The design matrix, the observed vector and the NNLS solution depend only on the apex
    // scan; the candidate id merely selects a column. Solving inside the per-candidate loop
    // above therefore re-probed every peak of a scan, and re-ran the NNLS, once for every
    // candidate that apexed in it -- on a wide-window run that is a second full pass over
    // the spectra. Grouping by resolved scan index collapses it to one solve per scan.
    //
    // Exactly the same numbers as the per-candidate version: `demix_solve_scan` reproduces
    // the assembly verbatim, `demix_features_for` reproduces the per-candidate reads, and
    // the group key is the same scan the old code resolved from `(apex_rt, precursor_mz)`.
    // Rows are patched by candidate id, so `results`' order is untouched.
    let mut results = results;
    if p.cfg.emit_demix_features {
        // Which candidates need a demix, grouped by the scan that serves them. BTreeMap so
        // the scan iteration order is deterministic.
        let mut by_scan: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for r in results.iter().flatten() {
            if r.peak_rank != 0 {
                continue;
            }
            let prec_mz = r.mz;
            if let Some(si) = demix_apex_scan(&scans, &rt_scan, r.apex_rt, prec_mz) {
                by_scan.entry(si).or_default().push(r.cid);
            }
        }
        let n_scans = by_scan.len();
        let n_cands: usize = by_scan.values().map(|v| v.len()).sum();
        let scan_jobs: Vec<(usize, Vec<u32>)> = by_scan.into_iter().collect();
        let solved: Vec<Vec<(u32, DemixFeatures)>> = scan_jobs
            .par_iter()
            .map(|(si, cids)| {
                let scan = &scans[*si];
                match demix_solve_scan(
                    fidx.as_ref(),
                    &lib,
                    scan,
                    &mass_off,
                    frag_tol,
                    scan.rt_seconds,
                    &rt_lo,
                    &rt_hi,
                    p.cfg,
                ) {
                    Some(d) => cids
                        .iter()
                        .map(|&cid| (cid, demix_features_for(&d, cid)))
                        .collect(),
                    None => Vec::new(),
                }
            })
            .collect();
        let feats: HashMap<u32, (f64, f64, f64, f64, f64)> = solved.into_iter().flatten().collect();
        info!(
            scans_solved = n_scans,
            candidates = n_cands,
            "extract: demix features (one NNLS per apex scan)"
        );
        for r in results.iter_mut().flatten() {
            if r.peak_rank != 0 {
                continue;
            }
            if let Some(&(expl, act, share, collin, shadow)) = feats.get(&r.cid) {
                r.deconv_explained = expl as f32;
                r.deconv_active = act as f32;
                r.deconv_share = share as f32;
                r.deconv_collin = collin as f32;
                r.deconv_shadow = shadow as f32;
            }
        }
    }

    // Append results in the deterministic cand_ids order (parallel work above was
    // order-preserving via `collect`), reproducing the serial push order exactly.
    let mut n_accepted = 0u64;
    // Top-K retained peaks (opt-in; empty for K=1).
    let (mut pk_cid, mut pk_rank): (Vec<u32>, Vec<i32>) = (Vec::new(), Vec::new());
    let (mut pk_apex, mut pk_start, mut pk_end): (Vec<f64>, Vec<f64>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut pk_ev, mut pk_area): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for r in results.into_iter().flatten() {
        n_accepted += 1;
        let rcid = r.cid;
        for (rank, apex, start, end, ev, area) in &r.peaks {
            pk_cid.push(rcid);
            pk_rank.push(*rank as i32);
            pk_apex.push(*apex);
            pk_start.push(*start);
            pk_end.push(*end);
            pk_ev.push(*ev);
            pk_area.push(*area);
        }
        cid_c.push(r.cid);
        peakrank_c.push(r.peak_rank as i32);
        apexrt_c.push(r.apex_rt);
        apexim_c.push(None);
        apexint_c.push(r.apex_int);
        nmatch_c.push(r.n_match);
        corun_c.push(r.corun);
        npred_c.push(r.npred);
        calrt_c.push(r.calrt);
        mz_c.push(r.mz);
        contested_c.push(r.contested);
        if p.cfg.emit_contested_features {
            contested_count_c.push(r.contested_count_frac);
            apportioned_c.push(r.apportioned_frac);
        }
        z_c.push(r.z);
        label_c.push(r.label);
        base_c.push(r.base);
        pform_c.push(r.pform);
        prot_c.push(r.prot);
        irt_c.push(r.irt);
        ms1_m1.push(r.ms1_m1);
        ms1_mono.push(r.ms1_mono);
        ms1_i1.push(r.ms1_i1);
        ms1_i2.push(r.ms1_i2);
        if p.cfg.emit_gate_diagnostics {
            gate_apex_c.push(r.gate_apex);
            gate_peakspec_c.push(r.gate_peak_spectral);
            gate_coel_c.push(r.gate_coelution);
            gate_se_c.push(r.gate_spectral_entropy);
        }
        if p.cfg.emit_demix_features {
            deconv_expl_c.push(r.deconv_explained);
            deconv_act_c.push(r.deconv_active);
            deconv_share_c.push(r.deconv_share);
            deconv_collin_c.push(r.deconv_collin);
            deconv_shadow_c.push(r.deconv_shadow);
        }
        for (cc, nm, fmz, omz, pint, rt, it) in r.chrom {
            ch_cid.push(cc);
            ch_name.push(nm);
            ch_fmz.push(fmz);
            ch_obsmz.push(omz);
            ch_pint.push(pint);
            ch_rt.push(rt);
            ch_int.push(it);
        }
    }

    let mut psms_cols = vec![
        Col::U32("candidate_id".into(), cid_c),
        Col::I32("peak_rank".into(), peakrank_c),
        Col::F64("apex_rt".into(), apexrt_c),
        Col::OptF64("apex_im".into(), apexim_c),
        Col::F32("apex_intensity".into(), apexint_c),
        Col::I32("n_matched_fragments".into(), nmatch_c),
        Col::I32("n_predicted_fragments".into(), npred_c),
        Col::I32("coelution_run".into(), corun_c),
        Col::F64("rt_pred_cal".into(), calrt_c),
        Col::F64("precursor_mz".into(), mz_c),
        Col::I32("charge".into(), z_c),
        Col::Str("label".into(), label_c),
        Col::U32("base_peptide_id".into(), base_c),
        Col::Str("peptidoform".into(), pform_c),
        Col::Str("protein".into(), prot_c),
        Col::F32("predicted_irt".into(), irt_c),
        Col::F64("contested_frac".into(), contested_c),
        Col::OptF64("ms1_isom1".into(), ms1_m1),
        Col::OptF64("ms1_mono".into(), ms1_mono),
        Col::OptF64("ms1_iso1".into(), ms1_i1),
        Col::OptF64("ms1_iso2".into(), ms1_i2),
    ];
    // Richer soft-competition columns only when emit_contested_features (default-off
    // keeps the schema byte-identical; contested_frac above is the pre-existing one).
    if p.cfg.emit_contested_features {
        psms_cols.push(Col::F64("contested_count_frac".into(), contested_count_c));
        psms_cols.push(Col::F64("apportioned_frac".into(), apportioned_c));
    }
    // Diagnostic gate-score columns only when enabled (default-off keeps the schema
    // byte-identical to the production chain).
    if p.cfg.emit_gate_diagnostics {
        psms_cols.push(Col::F32("gate_apex".into(), gate_apex_c));
        psms_cols.push(Col::F32("gate_peak_spectral".into(), gate_peakspec_c));
        psms_cols.push(Col::F32("gate_coelution".into(), gate_coel_c));
        psms_cols.push(Col::F32("gate_spectral_entropy".into(), gate_se_c));
    }
    if p.cfg.emit_demix_features {
        psms_cols.push(Col::F32("deconv_explained_frac".into(), deconv_expl_c));
        psms_cols.push(Col::F32("deconv_active".into(), deconv_act_c));
        psms_cols.push(Col::F32("deconv_share".into(), deconv_share_c));
        psms_cols.push(Col::F32("deconv_max_collinearity".into(), deconv_collin_c));
        psms_cols.push(Col::F32("shadow_kept_frac".into(), deconv_shadow_c));
    }
    let n_psms = write_table(p.out_psms, psms_cols)?;

    let n_chrom = write_table(
        p.out_chrom,
        vec![
            Col::U32("candidate_id".into(), ch_cid),
            Col::Str("frag_name".into(), ch_name),
            Col::F64("frag_mz".into(), ch_fmz),
            Col::F64("frag_obs_mz".into(), ch_obsmz),
            Col::F32("predicted_intensity".into(), ch_pint),
            // LargeList (64-bit offsets): the total chromatogram list-value count
            // can exceed the ~2.1B limit of a 32-bit ListArray offset buffer when
            // extraction accepts a very large candidate set (e.g. gates opened up).
            Col::LargeListF32("rt".into(), ch_rt),
            Col::LargeListF32("intensity".into(), ch_int),
        ],
    )?;

    // Top-K retained peaks (opt-in, sensitivity_plan P1.1/P1.2). Written next to
    // the psms table only when retain_top_peaks > 1; one row per (candidate, peak).
    if !pk_cid.is_empty() {
        let pk_path = format!("{}.peaks.parquet", p.out_psms);
        let n_peaks = write_table(
            &pk_path,
            vec![
                Col::U32("candidate_id".into(), pk_cid),
                Col::I32("peak_rank".into(), pk_rank),
                Col::F64("apex_rt".into(), pk_apex),
                Col::F64("start_rt".into(), pk_start),
                Col::F64("end_rt".into(), pk_end),
                Col::F64("evidence_count".into(), pk_ev),
                Col::F64("area".into(), pk_area),
            ],
        )?;
        info!(peaks = n_peaks, path = %pk_path, "extract: wrote top-K peak table");
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("accepted".to_string(), json!(n_accepted));
    stats.insert("scan_window".to_string(), json!(scan_window));
    for (path, schema, rows) in [
        (p.out_psms, artifact::PSMS_EXTRACTED, n_psms),
        (p.out_chrom, artifact::CHROMATOGRAMS, n_chrom),
    ] {
        ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "extract".to_string(),
            rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: json!({
                "frag_tol_ppm": p.cfg.frag_tol_ppm,
                "effective_frag_tol_ppm": frag_tol,
                "frag_ppm_offset": frag_offset,
                "presence_min_fragments": p.cfg.presence_min_fragments,
                "presence_min_coelution": p.cfg.presence_min_coelution,
                "gate_min_score": p.cfg.gate_min_score,
                "gate_mode": p.cfg.gate_mode,
                "gate_coelution_min": p.cfg.gate_coelution_min,
                "scan_window": scan_window,
            }),
            stats: stats.clone(),
            model_identity: None,
            elapsed_ms: elapsed,
        }
        .write_for(path)?;
    }

    info!(
        accepted = n_accepted,
        chromatograms = n_chrom,
        elapsed_ms = elapsed,
        "extract: done"
    );
    Ok((n_psms, n_chrom))
}

#[cfg(test)]
mod mass_offset_tests {
    use super::MassOffset;

    #[test]
    fn scalar_and_grid_interpolation() {
        // Scalar offset: constant factor regardless of m/z.
        let s = MassOffset {
            scalar_ppm: 5.0,
            grid_mz: vec![],
            grid_ppm: vec![],
        };
        assert!((s.factor_at(500.0) - (1.0 + 5e-6)).abs() < 1e-12);
        // Grid: linear interpolation between points, clamped past the ends.
        let g = MassOffset {
            scalar_ppm: 0.0,
            grid_mz: vec![200.0, 400.0, 600.0],
            grid_ppm: vec![2.0, 4.0, 0.0],
        };
        assert!((g.factor_at(300.0) - (1.0 + 3e-6)).abs() < 1e-12); // 200->400: 2->4, mid = 3
        assert!((g.factor_at(400.0) - (1.0 + 4e-6)).abs() < 1e-12); // exact grid point
        assert!((g.factor_at(100.0) - (1.0 + 2e-6)).abs() < 1e-12); // clamp low
        assert!((g.factor_at(900.0) - 1.0).abs() < 1e-12); // clamp high (ppm 0)
    }
}

#[cfg(test)]
mod coelution_tests {
    use super::{coelution_gate_score, peak_spectral_score};
    use std::collections::BTreeMap;

    fn g(rows: &[(f64, &[(u16, f32)])]) -> Vec<(f64, BTreeMap<u16, f32>)> {
        rows.iter()
            .map(|(rt, fs)| (*rt, fs.iter().cloned().collect()))
            .collect()
    }

    #[test]
    fn coeluting_fragments_score_high() {
        // frags 0,1,2 all peak together at group index 2
        let groups = g(&[
            (0.0, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
            (1.0, &[(0, 4.0), (1, 3.0), (2, 2.0)]),
            (2.0, &[(0, 9.0), (1, 8.0), (2, 5.0)]),
            (3.0, &[(0, 4.0), (1, 3.0), (2, 2.0)]),
            (4.0, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
        ]);
        let s = coelution_gate_score(&groups, &[0, 1, 2], &[0, 1], &[10.0, 8.0, 5.0]);
        assert!(s > 0.95, "co-eluting fragments should score high, got {s}");
    }

    #[test]
    fn non_coeluting_interferent_drops_the_score() {
        // frags 0,1 co-elute; frag 2 (a strong-predicted interferent) sits off-peak
        let groups = g(&[
            (0.0, &[(0, 1.0), (1, 1.0), (2, 9.0)]),
            (1.0, &[(0, 4.0), (1, 3.0), (2, 0.0)]),
            (2.0, &[(0, 9.0), (1, 8.0), (2, 0.0)]),
            (3.0, &[(0, 4.0), (1, 3.0), (2, 0.0)]),
            (4.0, &[(0, 1.0), (1, 1.0), (2, 0.0)]),
        ]);
        let s = coelution_gate_score(&groups, &[0, 1, 2], &[0, 1], &[10.0, 8.0, 9.0]);
        assert!(
            s < 0.8,
            "a strong non-co-eluting interferent should lower the score, got {s}"
        );
    }

    #[test]
    fn too_few_scans_does_not_reject() {
        let groups = g(&[(0.0, &[(0, 5.0)]), (1.0, &[(0, 9.0)])]);
        assert_eq!(coelution_gate_score(&groups, &[0], &[0], &[10.0]), 1.0);
    }

    #[test]
    fn peak_spectral_high_when_integrated_pattern_matches() {
        // observed peak-summed spectrum (9:8:5 at apex, tails scale) matches predicted
        let groups = g(&[
            (0.0, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
            (1.0, &[(0, 4.0), (1, 3.0), (2, 2.0)]),
            (2.0, &[(0, 9.0), (1, 8.0), (2, 5.0)]),
            (3.0, &[(0, 4.0), (1, 3.0), (2, 2.0)]),
            (4.0, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
        ]);
        let s = peak_spectral_score(&groups, &[0, 1], &[19.0, 16.0, 11.0]);
        assert!(s > 0.99, "integrated pattern matches predicted, got {s}");
    }

    #[test]
    fn peak_spectral_recovers_fragment_absent_at_apex_scan() {
        // A real strong-predicted fragment (2) is momentarily unsampled at the apex
        // scan (DIA scan gap) but present across the rest of the peak. The single-scan
        // apex Pearson would see obs=0 for it and collapse; integrating over the peak
        // recovers its true contribution and matches the predicted 19:16:16.
        let groups = g(&[
            (0.0, &[(0, 1.0), (1, 1.0), (2, 2.0)]),
            (1.0, &[(0, 4.0), (1, 3.0), (2, 6.0)]),
            (2.0, &[(0, 9.0), (1, 8.0), (2, 0.0)]), // frag 2 unsampled at apex scan
            (3.0, &[(0, 4.0), (1, 3.0), (2, 6.0)]),
            (4.0, &[(0, 1.0), (1, 1.0), (2, 2.0)]),
        ]);
        let s = peak_spectral_score(&groups, &[0, 1], &[19.0, 16.0, 16.0]);
        assert!(
            s > 0.99,
            "peak integration should recover the off-apex fragment, got {s}"
        );
    }
}
