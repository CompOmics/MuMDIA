//! Extended feature family: ion_series.
//!
//! Contract: `NAMES` and `values(&Evidence)` return the same number of items
//! in the same order. Names are globally unique, snake_case, and append-only
//! (they are part of the frozen extended feature schema). See the `Evidence`
//! struct in `stages/features.rs` for the available per-PSM evidence, and use
//! the shared kernels `crate::stats::{pearson, cosine, spectral_angle}` plus the
//! parent helpers `super::{mean, normalize_sum, best_xcorr, smooth3}`.
//!
//! b/y series coverage, ladder contiguity, complementarity, per-series
//! similarity and co-elution. `ordinal` and ion type come from the library.
//! Every value is finite: divisions are denominator-guarded, empty reductions
//! yield 0.0, and any non-finite intermediate is coerced to 0.0.
use super::Evidence;
use crate::stats::{cosine, pearson, spectral_angle};
use mumdia_core::constants::PROTON;
use std::collections::HashSet;

pub const NAMES: &[&str] = &[
    "n_matched_b",
    "n_matched_y",
    "frac_matched_b",
    "frac_matched_y",
    "by_count_balance",
    "by_intensity_ratio",
    "by_ratio_agreement",
    "by_ratio_consistency",
    "longest_b_run",
    "longest_y_run",
    "longest_run_max",
    "longest_run_frac_length",
    "series_coverage_b",
    "series_coverage_y",
    "sequence_coverage",
    "series_gap_fraction",
    "by_complement_count",
    "by_complement_mz_consistency",
    "by_complement_coelution",
    "ordinal_intensity_concordance_y",
    "ordinal_intensity_concordance_b",
    "series_coelution_y",
    "series_coelution_b",
    "spectral_angle_b",
    "spectral_angle_y",
    "pearson_b",
    "pearson_y",
    "cosine_charge1",
    "cosine_charge2",
    "charge_corr_balance",
    "mean_matched_ordinal_norm",
    "by_ion_contiguous_intensity",
    "by_ion_contiguous_lib_frac",
    "both_series_present",
];

const EPS: f64 = 1e-9;

/// Coerce any non-finite value (NaN/Inf) to 0.0.
#[inline]
fn fin(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[inline]
fn matched(e: &Evidence, i: usize) -> bool {
    e.obs_apex[i] > 0.0
}

/// Longest run of consecutive integers among a set of ordinals.
fn longest_consecutive(ords: &HashSet<u32>) -> u32 {
    if ords.is_empty() {
        return 0;
    }
    let mut sorted: Vec<u32> = ords.iter().copied().collect();
    sorted.sort_unstable();
    let mut best = 1u32;
    let mut cur = 1u32;
    for w in sorted.windows(2) {
        if w[1] == w[0] + 1 {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else if w[1] != w[0] {
            cur = 1;
        }
    }
    best
}

/// Mean pairwise population Pearson over the given peak traces (index order).
fn mean_pairwise_pearson(traces: &[&Vec<f64>]) -> f64 {
    let n = traces.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut cnt = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += fin(pearson(traces[i], traces[j]));
            cnt += 1.0;
        }
    }
    if cnt > 0.0 {
        fin(sum / cnt)
    } else {
        0.0
    }
}

/// Step-direction concordance between observed and predicted intensity along a
/// series ordered by ordinal. Fraction of adjacent pairs whose sign agrees.
fn ordinal_concordance(mut items: Vec<(u32, f64, f64)>) -> f64 {
    // items: (ordinal, obs, pred), only matched fragments of one series.
    if items.len() < 2 {
        return 0.0;
    }
    items.sort_by_key(|a| a.0);
    let sign = |x: f64| -> i32 {
        if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        }
    };
    let mut agree = 0.0;
    let mut total = 0.0;
    for w in items.windows(2) {
        let os = sign(w[1].1 - w[0].1);
        let ps = sign(w[1].2 - w[0].2);
        if os == ps {
            agree += 1.0;
        }
        total += 1.0;
    }
    if total > 0.0 {
        agree / total
    } else {
        0.0
    }
}

pub fn values(e: &Evidence) -> Vec<f64> {
    let k = e.pred.len();
    let lf = e.seq_len as f64;
    let lm1 = lf - 1.0; // backbone cleavage sites
    let l_i = e.seq_len as i64;

    // Series index partitions.
    let b_idx: Vec<usize> = (0..k).filter(|&i| e.is_b[i]).collect();
    let y_idx: Vec<usize> = (0..k).filter(|&i| !e.is_b[i]).collect();

    // Matched counts per series.
    let n_matched_b = b_idx.iter().filter(|&&i| matched(e, i)).count() as f64;
    let n_matched_y = y_idx.iter().filter(|&&i| matched(e, i)).count() as f64;
    let n_pred_b = b_idx.len() as f64;
    let n_pred_y = y_idx.len() as f64;

    let frac_matched_b = if n_pred_b > 0.0 {
        n_matched_b / n_pred_b
    } else {
        0.0
    };
    let frac_matched_y = if n_pred_y > 0.0 {
        n_matched_y / n_pred_y
    } else {
        0.0
    };

    let by_count_balance = {
        let mx = n_matched_b.max(n_matched_y);
        if mx > 0.0 {
            n_matched_b.min(n_matched_y) / mx
        } else {
            0.0
        }
    };

    // Series apex-intensity sums (observed) and predicted sums.
    let bo: f64 = b_idx.iter().map(|&i| e.obs_apex[i].max(0.0)).sum();
    let yo: f64 = y_idx.iter().map(|&i| e.obs_apex[i].max(0.0)).sum();
    let bp: f64 = b_idx.iter().map(|&i| e.pred[i].max(0.0)).sum();
    let yp: f64 = y_idx.iter().map(|&i| e.pred[i].max(0.0)).sum();

    let by_intensity_ratio = {
        let den = bo + yo + EPS;
        fin(yo / den)
    };

    let by_ratio_agreement = {
        // Bounded observed-vs-predicted b/y LOG-ODDS discrepancy, squashed to
        // [0,1) by tanh. This is the log-odds analog (tail-sensitive), distinct
        // from the linear-fraction `by_ratio_consistency` below; the earlier
        // raw log-ratio version saturated to ~35 for single-series peptides.
        let lo = ((bo + EPS) / (yo + EPS)).ln();
        let lp = ((bp + EPS) / (yp + EPS)).ln();
        fin((0.5 * (lo - lp)).tanh().abs())
    };

    // Both fragment series observed: encodes single-series as a category rather
    // than letting the ratio features carry it as a saturated magnitude.
    let both_series_present = if n_matched_b > 0.0 && n_matched_y > 0.0 {
        1.0
    } else {
        0.0
    };

    let by_ratio_consistency = {
        let fo = {
            let d = bo + yo;
            if d > 0.0 {
                bo / d
            } else {
                0.0
            }
        };
        let fp = {
            let d = bp + yp;
            if d > 0.0 {
                bp / d
            } else {
                0.0
            }
        };
        (fo - fp).abs()
    };

    // Matched ordinal sets per series (for runs, coverage, gaps).
    let mut b_ord_matched: HashSet<u32> = HashSet::new();
    let mut y_ord_matched: HashSet<u32> = HashSet::new();
    for &i in &b_idx {
        if matched(e, i) {
            b_ord_matched.insert(e.ordinal[i]);
        }
    }
    for &i in &y_idx {
        if matched(e, i) {
            y_ord_matched.insert(e.ordinal[i]);
        }
    }

    let longest_b_run = longest_consecutive(&b_ord_matched) as f64;
    let longest_y_run = longest_consecutive(&y_ord_matched) as f64;
    let longest_run_max = longest_b_run.max(longest_y_run);
    let longest_run_frac_length = if lm1 > 0.0 {
        longest_run_max / lm1
    } else {
        0.0
    };

    let series_coverage_b = if lm1 > 0.0 {
        b_ord_matched.len() as f64 / lm1
    } else {
        0.0
    };
    let series_coverage_y = if lm1 > 0.0 {
        y_ord_matched.len() as f64 / lm1
    } else {
        0.0
    };

    // sequence_coverage: union of covered cleavage sites (b_k -> site k,
    // y_k -> site L-k), restricted to interior sites 1..=L-1.
    let sequence_coverage = {
        let mut sites: HashSet<i64> = HashSet::new();
        for &o in &b_ord_matched {
            let s = o as i64;
            if s >= 1 && s < l_i {
                sites.insert(s);
            }
        }
        for &o in &y_ord_matched {
            let s = l_i - o as i64;
            if s >= 1 && s < l_i {
                sites.insert(s);
            }
        }
        if lm1 > 0.0 {
            sites.len() as f64 / lm1
        } else {
            0.0
        }
    };

    // series_gap_fraction: over the union of matched ordinals, fraction of
    // positions in [min,max] that are not matched.
    let series_gap_fraction = {
        let mut all: HashSet<u32> = HashSet::new();
        all.extend(b_ord_matched.iter().copied());
        all.extend(y_ord_matched.iter().copied());
        if all.len() < 2 {
            0.0
        } else {
            let mn = *all.iter().min().unwrap();
            let mx = *all.iter().max().unwrap();
            let span = (mx - mn) as f64 + 1.0; // positions in the covered range
            let present = all.len() as f64;
            if span > 0.0 {
                fin((span - present) / span)
            } else {
                0.0
            }
        }
    };

    // Complementarity: pair b_k with y_{L-k}.
    // Representative matched y index per ordinal (index order keeps determinism).
    let mut by_complement_count = 0.0;
    for &kord in &b_ord_matched {
        let comp = l_i - kord as i64;
        if comp >= 1
            && comp < l_i
            && comp <= u32::MAX as i64
            && y_ord_matched.contains(&(comp as u32))
        {
            by_complement_count += 1.0;
        }
    }

    // by_complement_mz_consistency: for matched, singly-charged complementary
    // pairs, ppm deviation of obs_mz(b)+obs_mz(y) from M + 2*proton.
    let by_complement_mz_consistency = {
        let mut acc = 0.0;
        let mut cnt = 0.0;
        if e.charge > 0 {
            let neutral = e.precursor_mz * e.charge as f64 - e.charge as f64 * PROTON;
            let target = neutral + 2.0 * PROTON;
            if target > 0.0 {
                // First matched, charge-1, obs_mz>0 y index per ordinal.
                for &bi in &b_idx {
                    if !matched(e, bi) || e.frag_charge[bi] != 1 || e.frag_obs_mz[bi] <= 0.0 {
                        continue;
                    }
                    let comp = l_i - e.ordinal[bi] as i64;
                    if comp < 1 || comp > l_i - 1 || comp > u32::MAX as i64 {
                        continue;
                    }
                    let comp_u = comp as u32;
                    for &yi in &y_idx {
                        if e.ordinal[yi] == comp_u
                            && matched(e, yi)
                            && e.frag_charge[yi] == 1
                            && e.frag_obs_mz[yi] > 0.0
                        {
                            let s = e.frag_obs_mz[bi] + e.frag_obs_mz[yi];
                            let ppm = (s - target).abs() / target * 1e6;
                            acc += fin(ppm);
                            cnt += 1.0;
                            break;
                        }
                    }
                }
            }
        }
        if cnt > 0.0 {
            fin(acc / cnt)
        } else {
            0.0
        }
    };

    // by_complement_coelution: mean pearson over matched complementary pairs'
    // peak traces.
    let by_complement_coelution = {
        let mut acc = 0.0;
        let mut cnt = 0.0;
        let has_traces = e.traces.len() == k;
        if has_traces {
            for &bi in &b_idx {
                if !matched(e, bi) {
                    continue;
                }
                let comp = l_i - e.ordinal[bi] as i64;
                if comp < 1 || comp > l_i - 1 || comp > u32::MAX as i64 {
                    continue;
                }
                let comp_u = comp as u32;
                for &yi in &y_idx {
                    if e.ordinal[yi] == comp_u && matched(e, yi) {
                        acc += fin(pearson(&e.traces[bi], &e.traces[yi]));
                        cnt += 1.0;
                        break;
                    }
                }
            }
        }
        if cnt > 0.0 {
            fin(acc / cnt)
        } else {
            0.0
        }
    };

    // ordinal_intensity_concordance (matched fragments per series).
    let ordinal_intensity_concordance_y = {
        let items: Vec<(u32, f64, f64)> = y_idx
            .iter()
            .filter(|&&i| matched(e, i))
            .map(|&i| (e.ordinal[i], e.obs_apex[i], e.pred[i]))
            .collect();
        ordinal_concordance(items)
    };
    let ordinal_intensity_concordance_b = {
        let items: Vec<(u32, f64, f64)> = b_idx
            .iter()
            .filter(|&&i| matched(e, i))
            .map(|&i| (e.ordinal[i], e.obs_apex[i], e.pred[i]))
            .collect();
        ordinal_concordance(items)
    };

    // Per-series co-elution: mean pairwise pearson among matched peak traces.
    let has_traces = e.traces.len() == k;
    let series_coelution_y = if has_traces {
        let t: Vec<&Vec<f64>> = y_idx
            .iter()
            .filter(|&&i| matched(e, i))
            .map(|&i| &e.traces[i])
            .collect();
        mean_pairwise_pearson(&t)
    } else {
        0.0
    };
    let series_coelution_b = if has_traces {
        let t: Vec<&Vec<f64>> = b_idx
            .iter()
            .filter(|&&i| matched(e, i))
            .map(|&i| &e.traces[i])
            .collect();
        mean_pairwise_pearson(&t)
    } else {
        0.0
    };

    // Per-series library-vs-observed spectral similarity (all fragments).
    let b_lib: Vec<f64> = b_idx.iter().map(|&i| e.pred[i]).collect();
    let b_obs: Vec<f64> = b_idx.iter().map(|&i| e.obs_apex[i]).collect();
    let y_lib: Vec<f64> = y_idx.iter().map(|&i| e.pred[i]).collect();
    let y_obs: Vec<f64> = y_idx.iter().map(|&i| e.obs_apex[i]).collect();

    let spectral_angle_b = fin(spectral_angle(&b_lib, &b_obs));
    let spectral_angle_y = fin(spectral_angle(&y_lib, &y_obs));
    let pearson_b = fin(pearson(&b_lib, &b_obs));
    let pearson_y = fin(pearson(&y_lib, &y_obs));

    // Charge-resolved cosine similarity.
    let c1_idx: Vec<usize> = (0..k).filter(|&i| e.frag_charge[i] == 1).collect();
    let c2_idx: Vec<usize> = (0..k).filter(|&i| e.frag_charge[i] >= 2).collect();
    let cosine_charge1 = {
        let lib: Vec<f64> = c1_idx.iter().map(|&i| e.pred[i]).collect();
        let obs: Vec<f64> = c1_idx.iter().map(|&i| e.obs_apex[i]).collect();
        fin(cosine(&lib, &obs))
    };
    let cosine_charge2 = if c2_idx.len() >= 2 {
        let lib: Vec<f64> = c2_idx.iter().map(|&i| e.pred[i]).collect();
        let obs: Vec<f64> = c2_idx.iter().map(|&i| e.obs_apex[i]).collect();
        fin(cosine(&lib, &obs))
    } else {
        0.0
    };

    // charge_corr_balance: min/max of mean ref-profile correlation per charge.
    let charge_corr_balance = if has_traces && e.ref_profile.len() == e.axis.len() {
        let mean_refcorr = |idx: &[usize]| -> f64 {
            let mut s = 0.0;
            let mut c = 0.0;
            for &i in idx {
                s += fin(pearson(&e.traces[i], &e.ref_profile));
                c += 1.0;
            }
            if c > 0.0 {
                s / c
            } else {
                0.0
            }
        };
        let m1 = mean_refcorr(&c1_idx);
        let m2 = mean_refcorr(&c2_idx);
        let mx = m1.max(m2);
        let mn = m1.min(m2);
        if mx.abs() > EPS {
            fin(mn / mx)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // mean_matched_ordinal_norm.
    let mean_matched_ordinal_norm = {
        let mut s = 0.0;
        let mut c = 0.0;
        for i in 0..k {
            if matched(e, i) {
                s += e.ordinal[i] as f64;
                c += 1.0;
            }
        }
        if c > 0.0 && lf > 0.0 {
            fin((s / c) / lf)
        } else {
            0.0
        }
    };

    // Contiguous ladder members: matched fragments having a matched same-series
    // neighbor (ordinal +/- 1). Sum apex (log) and predicted-intensity fraction.
    let mut contig_apex = 0.0;
    let mut contig_lib = 0.0;
    let total_lib: f64 = e.pred.iter().map(|v| v.max(0.0)).sum();
    for i in 0..k {
        if !matched(e, i) {
            continue;
        }
        let ord = e.ordinal[i];
        let set = if e.is_b[i] {
            &b_ord_matched
        } else {
            &y_ord_matched
        };
        let has_neighbor = (ord > 0 && set.contains(&(ord - 1))) || set.contains(&(ord + 1));
        if has_neighbor {
            contig_apex += e.obs_apex[i].max(0.0);
            contig_lib += e.pred[i].max(0.0);
        }
    }
    let by_ion_contiguous_intensity = fin((1.0 + contig_apex).ln());
    let by_ion_contiguous_lib_frac = if total_lib > 0.0 {
        fin(contig_lib / total_lib)
    } else {
        0.0
    };

    let out = vec![
        n_matched_b,
        n_matched_y,
        frac_matched_b,
        frac_matched_y,
        by_count_balance,
        by_intensity_ratio,
        by_ratio_agreement,
        by_ratio_consistency,
        longest_b_run,
        longest_y_run,
        longest_run_max,
        longest_run_frac_length,
        series_coverage_b,
        series_coverage_y,
        sequence_coverage,
        series_gap_fraction,
        by_complement_count,
        by_complement_mz_consistency,
        by_complement_coelution,
        ordinal_intensity_concordance_y,
        ordinal_intensity_concordance_b,
        series_coelution_y,
        series_coelution_b,
        spectral_angle_b,
        spectral_angle_y,
        pearson_b,
        pearson_y,
        cosine_charge1,
        cosine_charge2,
        charge_corr_balance,
        mean_matched_ordinal_norm,
        by_ion_contiguous_intensity,
        by_ion_contiguous_lib_frac,
        both_series_present,
    ];
    debug_assert_eq!(out.len(), NAMES.len());
    out.into_iter().map(fin).collect()
}
