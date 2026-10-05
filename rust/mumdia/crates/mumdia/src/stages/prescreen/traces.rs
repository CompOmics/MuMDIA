//! Fragment traces over consecutive same-window scans (the prototype's `fragment_features`).
//!
//! A pool is `scans` consecutive spectra of one isolation window (non-overlapping, in RT
//! order). Merged pooling sorts the pool's positive peaks and closes a cluster as soon as the
//! next peak would make `max - min` exceed `cluster_da`, so a chain of near neighbours cannot
//! grow wider than the bound. A cluster is kept when detected in at least `min_detections`
//! scans, as an intensity-weighted centroid with its summed intensity, detection count and an
//! L2-normalised per-scan intensity profile. Unmerged pooling keeps every pooled peak as its own
//! one-scan feature, for comparison. There is no feature-count cap; the uncapped single-scan
//! branch (the core score) is unaffected.

use std::collections::HashMap;

use mumdia_core::config::{PrescreenPooling, PrescreenTraceConfig};

use super::tags::{fit_weight, nearest, path_rms, Alphabet, Graph};

pub struct Pool {
    pub mean_rt: f64,
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
    pub detections: Vec<u8>,
    /// `profile[i * scans..(i + 1) * scans]`, unit L2 norm.
    pub profile: Vec<f32>,
    pub scans: usize,
}

impl Pool {
    pub fn trace(&self, i: usize) -> &[f32] {
        &self.profile[i * self.scans..(i + 1) * self.scans]
    }
}

/// Pool consecutive spectra: `spectra[t]` is `(rt, positive peaks (mz, intensity) ascending)`.
pub fn pool(spectra: &[(f64, Vec<(f64, f64)>)], cfg: &PrescreenTraceConfig) -> Pool {
    let w = cfg.scans;
    let mut peaks: Vec<(f64, f64, usize)> = Vec::new();
    for (t, (_, p)) in spectra.iter().enumerate() {
        peaks.extend(p.iter().filter(|x| x.1 > 0.0).map(|&(m, i)| (m, i, t)));
    }
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    let mean_rt = spectra.iter().map(|s| s.0).sum::<f64>() / spectra.len().max(1) as f64;
    let mut out = Pool {
        mean_rt,
        mz: Vec::new(),
        intensity: Vec::new(),
        detections: Vec::new(),
        profile: Vec::new(),
        scans: w,
    };
    let push = |members: &[(f64, f64, usize)], out: &mut Pool| {
        let mut trace = vec![0.0f64; w];
        let (mut sm, mut si) = (0.0, 0.0);
        for &(m, i, t) in members {
            trace[t] += i;
            sm += m * i;
            si += i;
        }
        let seen = trace.iter().filter(|&&x| x > 0.0).count();
        let need = match cfg.pooling {
            PrescreenPooling::Merged => cfg.min_detections,
            PrescreenPooling::Unmerged => 1,
        };
        if seen < need || si <= 0.0 {
            return;
        }
        let norm = trace.iter().map(|x| x * x).sum::<f64>().sqrt();
        out.mz.push(sm / si);
        out.intensity.push(si);
        out.detections.push(seen as u8);
        out.profile.extend(trace.iter().map(|x| (x / norm) as f32));
    };
    match cfg.pooling {
        PrescreenPooling::Unmerged => {
            for k in 0..peaks.len() {
                push(&peaks[k..k + 1], &mut out);
            }
        }
        PrescreenPooling::Merged => {
            let mut i = 0;
            while i < peaks.len() {
                let mut j = i + 1;
                while j < peaks.len() && peaks[j].0 - peaks[i].0 <= cfg.cluster_da {
                    j += 1;
                }
                push(&peaks[i..j], &mut out);
                i = j;
            }
        }
    }
    // Clusters are emitted in ascending start mass; centroids can reorder only within the
    // bound, so sort to keep the m/z array ascending.
    let mut order: Vec<usize> = (0..out.mz.len()).collect();
    order.sort_by(|&a, &b| out.mz[a].total_cmp(&out.mz[b]).then(a.cmp(&b)));
    Pool {
        mean_rt,
        mz: order.iter().map(|&i| out.mz[i]).collect(),
        intensity: order.iter().map(|&i| out.intensity[i]).collect(),
        detections: order.iter().map(|&i| out.detections[i]).collect(),
        profile: order.iter().flat_map(|&i| out.trace(i).to_vec()).collect(),
        scans: w,
    }
}

/// Uncentred cosine of two unit traces, clipped to [0, 1].
pub fn quality(a: &[f32], b: &[f32]) -> f64 {
    let q: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    q.clamp(0.0, 1.0)
}

/// Per key, the best `min pairwise trace cosine of the four path features x soft fit` in a pool.
pub fn pool_tag_coherence(
    p: &Pool,
    alpha: &Alphabet,
    tol: f64,
    sigma: f64,
    max_charge: i32,
) -> HashMap<u32, f64> {
    let mut best: HashMap<u32, f64> = HashMap::new();
    let m = |s: u32| alpha.masses[s as usize];
    for z in 1..=max_charge {
        let g = Graph::build(&p.mz, z, tol, alpha, None);
        for i in 0..p.mz.len() {
            for e in g.edges(i) {
                let (j, a) = (g.target[e] as usize, g.label[e]);
                for f in g.edges(j) {
                    let (k, b) = (g.target[f] as usize, g.label[f]);
                    for h in g.edges(k) {
                        let (l, c) = (g.target[h] as usize, g.label[h]);
                        let nodes = [i, j, k, l];
                        let mut coh = 1.0f64;
                        for x in 0..4 {
                            for y in x + 1..4 {
                                coh = coh.min(quality(p.trace(nodes[x]), p.trace(nodes[y])));
                            }
                        }
                        let cum = [m(a), m(a) + m(b), m(a) + m(b) + m(c)];
                        let q = coh * fit_weight(path_rms(&p.mz, &nodes, &cum, z), sigma);
                        let key =
                            alpha.key(alpha.canonical(a as u16, b as u16, c as u16), z, false);
                        let e = best.entry(key).or_insert(0.0);
                        if q > *e {
                            *e = q;
                        }
                    }
                }
            }
        }
    }
    best
}

/// Tag-coherence trace score of one candidate in one pool: the sum over its distinct trimer
/// codes of the best coherence at an allowed charge.
pub fn candidate_coherence(keys_by_code: &[Vec<u32>], pool: &HashMap<u32, f64>) -> f64 {
    keys_by_code
        .iter()
        .map(|keys| {
            keys.iter()
                .filter_map(|k| pool.get(k))
                .fold(0.0, |a: f64, &b| a.max(b))
        })
        .sum()
}

/// Coherent-fragment trace score: the candidate's fragments matched to pool features (nearest
/// within `tol`); the most intense matched feature anchors the trace; every distinct matched
/// feature whose cosine with it reaches `min_quality` counts with its rarity weight, so a
/// ladder need not be connected to count. Divided by `sqrt(L)`.
pub fn coherent_fragments(
    frag: &[f64],
    p: &Pool,
    weight: impl Fn(f64) -> f64,
    tol: f64,
    min_quality: f64,
    sqrt_l: f64,
) -> f64 {
    let mut hit: Vec<usize> = frag
        .iter()
        .filter_map(|&x| nearest(&p.mz, x, tol))
        .collect();
    hit.sort_unstable();
    hit.dedup();
    let Some(&anchor) = hit
        .iter()
        .max_by(|&&a, &&b| p.intensity[a].total_cmp(&p.intensity[b]).then(b.cmp(&a)))
    else {
        return 0.0;
    };
    hit.iter()
        .filter(|&&i| quality(p.trace(i), p.trace(anchor)) >= min_quality)
        .map(|&i| weight(p.mz[i]))
        .sum::<f64>()
        / sqrt_l
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(pooling: PrescreenPooling) -> PrescreenTraceConfig {
        PrescreenTraceConfig {
            pooling,
            ..PrescreenTraceConfig::default()
        }
    }

    /// A chain of near neighbours 0.004 apart cannot grow one cluster wider than 0.01; a
    /// cluster seen in one scan only is dropped by merged pooling and kept by unmerged pooling.
    #[test]
    fn merged_clusters_are_bounded_and_need_two_detections() {
        let spectra: Vec<(f64, Vec<(f64, f64)>)> = (0..7)
            .map(|t| {
                let mut p = vec![(500.0 + 0.004 * t as f64, 10.0)];
                if t == 3 {
                    p.push((800.0, 5.0));
                }
                (t as f64, p)
            })
            .collect();
        let merged = pool(&spectra, &cfg(PrescreenPooling::Merged));
        // 500.000-500.008 and 500.012-500.020 form two bounded clusters; 500.024 alone is dropped.
        assert_eq!(
            merged.mz.len(),
            2,
            "the 0.024 Da chain splits: {:?}",
            merged.mz
        );
        assert!(
            merged.mz.iter().all(|&m| m < 700.0),
            "single detection dropped"
        );
        for i in 0..merged.mz.len() {
            let n: f64 = merged.trace(i).iter().map(|&x| x as f64 * x as f64).sum();
            assert!((n - 1.0).abs() < 1e-5);
            assert!(merged.detections[i] >= 2);
        }
        let unmerged = pool(&spectra, &cfg(PrescreenPooling::Unmerged));
        assert_eq!(unmerged.mz.len(), 8);
        assert!(unmerged.mz.contains(&800.0));
    }

    #[test]
    fn centroids_are_intensity_weighted_and_intensities_summed() {
        let spectra = vec![(0.0, vec![(500.000, 1.0)]), (1.0, vec![(500.004, 3.0)])];
        let p = pool(&spectra, &cfg(PrescreenPooling::Merged));
        assert_eq!(p.mz.len(), 1);
        assert!((p.mz[0] - 500.003).abs() < 1e-9);
        assert_eq!(p.intensity[0], 4.0);
        assert_eq!(p.detections[0], 2);
    }

    #[test]
    fn coherent_fragments_count_compatible_traces_without_a_connected_ladder() {
        let up = [1.0f64, 2.0, 4.0, 2.0, 1.0, 0.0, 0.0];
        let other = [0.0f64, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0];
        let mk = |m: f64, tr: &[f64]| -> Vec<(usize, f64, f64)> {
            tr.iter()
                .enumerate()
                .filter(|x| *x.1 > 0.0)
                .map(|(t, &i)| (t, m, i))
                .collect()
        };
        let mut per: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 7];
        for (t, m, i) in mk(400.0, &up)
            .into_iter()
            .chain(mk(650.0, &up))
            .chain(mk(900.0, &other))
        {
            per[t].push((m, i));
        }
        let spectra: Vec<(f64, Vec<(f64, f64)>)> = per
            .into_iter()
            .enumerate()
            .map(|(t, p)| (t as f64, p))
            .collect();
        let p = pool(&spectra, &cfg(PrescreenPooling::Merged));
        let s = coherent_fragments(&[400.0, 650.0, 900.0], &p, |_| 1.0, 0.005, 0.6, 1.0);
        assert_eq!(s, 2.0, "400 and 650 share a trace; 900 elutes elsewhere");
    }
}
