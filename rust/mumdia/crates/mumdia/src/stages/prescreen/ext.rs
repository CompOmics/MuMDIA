//! The optional components, run window group by window group so only one group's tags, pools
//! and frequencies are in memory at a time. Nothing here runs under the default configuration.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use mumdia_core::config::{Config, PrescreenRetrieval, PrescreenTraceScore};
use rayon::prelude::*;

use super::evidence::{self, CandidateTags, WindowFreq, N_COMPONENTS};
use super::masses::{self, Hypothesis};
use super::retrieval::{self, BackboneIndex, ObservedIndex};
use super::tags::{self, Alphabet, SpectrumTags};
use super::traces;
use super::{fragment_mz, Candidate, Histogram, Spectra};

/// Component columns beyond the tag evidence ones.
pub const TRACE: usize = N_COMPONENTS;
pub const MASS: usize = N_COMPONENTS + 1;
pub const N_EXT: usize = N_COMPONENTS + 2;
pub const NAMES: [&str; N_EXT] = ["tag", "fasta", "complement", "flank", "trace", "mass"];

/// What the extended pass produced.
pub struct Extended {
    /// Retrieval outcome per candidate; `None` when `retrieval = "all"`.
    pub retrieved: Option<Vec<bool>>,
    pub components: Vec<[f64; N_EXT]>,
    pub stats: serde_json::Map<String, serde_json::Value>,
    pub hypotheses: Vec<(u32, f64, Hypothesis, u8)>,
}

/// Per candidate: residue states, keys and family, when the tag machinery is needed.
pub struct TagView {
    pub states: Vec<Option<Vec<Option<u16>>>>,
    pub keys: Vec<Vec<u32>>,
    pub norm_keys: Vec<Vec<u32>>,
    pub family: Vec<u32>,
    pub families: Vec<Vec<u16>>,
    pub unsupported: usize,
}

pub fn needs_tags(cfg: &Config) -> bool {
    let p = &cfg.prescreen;
    p.retrieval == PrescreenRetrieval::Tags
        || p.complement_bonus > 0.0
        || p.tag_bonus > 0.0
        || p.fasta_bonus > 0.0
        || p.flank_bonus > 0.0
        || p.trace.enabled
        || p.mass_hypotheses.enabled
}

pub fn tag_view(pforms: &[&str], charge: &[i32], alpha: &Alphabet, cfg: &Config) -> TagView {
    let p = &cfg.prescreen;
    let gapped = p.tags.gap_edges;
    let states: Vec<Option<Vec<Option<u16>>>> =
        pforms.par_iter().map(|pf| alpha.tokenise(pf)).collect();
    let zmax = |i: usize| {
        p.tags
            .max_charge
            .min(p.max_frag_charge)
            .min(charge[i])
            .max(1)
    };
    let keys: Vec<Vec<u32>> = (0..states.len())
        .into_par_iter()
        .map(|i| match &states[i] {
            Some(s) => retrieval::candidate_keys(s, zmax(i), gapped, alpha),
            None => Vec::new(),
        })
        .collect();
    let norm_keys: Vec<Vec<u32>> = (0..states.len())
        .into_par_iter()
        .map(|i| match &states[i] {
            Some(s) => retrieval::backbone_keys(s, zmax(i), gapped, alpha),
            None => Vec::new(),
        })
        .collect();
    let mut fam_of: HashMap<Vec<u16>, u32> = HashMap::new();
    let mut families: Vec<Vec<u16>> = Vec::new();
    let family = states
        .iter()
        .map(|s| {
            let b: Vec<u16> = s
                .as_ref()
                .map(|v| {
                    v.iter()
                        .map(|x| x.map(|y| alpha.normal[y as usize]).unwrap_or(u16::MAX))
                        .collect()
                })
                .unwrap_or_default();
            *fam_of.entry(b.clone()).or_insert_with(|| {
                families.push(b);
                (families.len() - 1) as u32
            })
        })
        .collect();
    let unsupported = keys.iter().filter(|k| k.is_empty()).count();
    TagView {
        states,
        keys,
        norm_keys,
        family,
        families,
        unsupported,
    }
}

/// Seeded sample of spectra for mass hypotheses (`sample_spectra = 0`: all).
fn sampled(n: usize, k: usize, seed: u64) -> Vec<bool> {
    if k == 0 || k >= n {
        return vec![true; n];
    }
    let mut idx: Vec<(u64, usize)> = (0..n)
        .map(|i| {
            let mut z = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31), i)
        })
        .collect();
    idx.sort_unstable();
    let mut out = vec![false; n];
    for &(_, i) in &idx[..k] {
        out[i] = true;
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    sp: &Spectra,
    cands: &[Option<Candidate>],
    members: &[Vec<u32>],
    view: &TagView,
    alpha: &Alphabet,
    cfg: &Config,
    ms1: Option<&[masses::Ms1Scan]>,
    masses_of: &(dyn Fn(usize) -> Vec<f64> + Sync),
) -> Extended {
    let p = &cfg.prescreen;
    let n = cands.len();
    let tol = p.tags.edge_tol_da;
    let sigma = p.tags.rms_sigma_da;
    let pairs = p.tags.gap_edges.then(|| tags::pair_steps(alpha));
    let want = evidence::Wanted {
        tag: p.tag_bonus > 0.0,
        fasta: p.fasta_bonus > 0.0,
        complement: p.complement_bonus > 0.0,
        flank: p.flank_bonus > 0.0,
    };
    let index = want
        .fasta
        .then(|| BackboneIndex::build(&view.families, alpha));
    let tag_retrieval = p.retrieval == PrescreenRetrieval::Tags;
    let need_spectrum_tags = tag_retrieval || want.any();
    let mut retrieved = vec![false; n];
    let mut components = vec![[0.0f64; N_EXT]; n];
    let mut hyps_out = Vec::new();
    let paths = AtomicU64::new(0);
    let tag_records = AtomicU64::new(0);
    let mut examined = 0u64;
    let mut skipped_by_family = 0u64;
    let mut index_keys = 0u64;
    let sample = sampled(
        sp.rt.len(),
        p.mass_hypotheses.sample_spectra,
        cfg.prescreen.seed,
    );

    for (gi, g) in sp.groups.iter().enumerate() {
        if members[gi].is_empty() {
            continue;
        }
        let range = g.first..g.first + g.count;
        let rts = &sp.rt[range.clone()];
        // Positive peaks as f64, the tag and evidence input.
        let pos: Vec<Vec<f64>> = range
            .clone()
            .into_par_iter()
            .map(|s| {
                let (a, b) = (sp.offsets[s], sp.offsets[s + 1]);
                (a..b)
                    .filter(|&k| sp.positive[k])
                    .map(|k| sp.mz[k] as f64)
                    .collect()
            })
            .collect();
        let spec_tags: Vec<SpectrumTags> = if need_spectrum_tags {
            pos.par_iter()
                .map(|mz| {
                    let t = tags::discover(mz, alpha, tol, p.tags.max_charge, pairs.as_deref());
                    paths.fetch_add(t.paths, Ordering::Relaxed);
                    tag_records.fetch_add(t.keys.len() as u64, Ordering::Relaxed);
                    t
                })
                .collect()
        } else {
            Vec::new()
        };
        let span = |c: &Candidate| match c.rt {
            Some((lo, hi)) => (
                rts.partition_point(|&r| r < lo),
                rts.partition_point(|&r| r <= hi),
            ),
            None => (0, rts.len()),
        };

        // ---- retrieval (indexed), optionally family-first ----
        if tag_retrieval {
            let obs = ObservedIndex::build(&spec_tags, rts, None);
            index_keys += obs.len() as u64;
            let fam_hit: Option<std::collections::HashSet<u32>> = p.delayed_modforms.then(|| {
                let norm = ObservedIndex::build(&spec_tags, rts, Some(alpha));
                members[gi]
                    .iter()
                    .filter_map(|&i| {
                        let c = cands[i as usize].as_ref()?;
                        norm.any(&view.norm_keys[i as usize], c.rt)
                            .then_some(view.family[i as usize])
                    })
                    .collect()
            });
            for &i in &members[gi] {
                let i = i as usize;
                let Some(c) = cands[i].as_ref() else { continue };
                if view.keys[i].is_empty() {
                    // No trimer the alphabet can express: retrieval cannot judge it.
                    retrieved[i] = true;
                    continue;
                }
                if let Some(h) = &fam_hit {
                    if !h.contains(&view.family[i]) {
                        skipped_by_family += 1;
                        continue;
                    }
                }
                examined += 1;
                if obs.any(&view.keys[i], c.rt) {
                    retrieved[i] = true;
                }
            }
        }

        // ---- tag evidence components ----
        if want.any() {
            let freq = WindowFreq::build(&spec_tags);
            let params = evidence::Params {
                alpha,
                tol,
                sigma,
                index: index.as_ref(),
            };
            let got: Vec<[f64; N_COMPONENTS]> = members[gi]
                .par_iter()
                .map(|&i| {
                    let i = i as usize;
                    let (Some(c), Some(states)) = (cands[i].as_ref(), view.states[i].as_ref())
                    else {
                        return [0.0; N_COMPONENTS];
                    };
                    let cm = masses_of(i);
                    let ct = CandidateTags {
                        states,
                        masses: &cm,
                        zmax: p.tags.max_charge.min(c.nz),
                    };
                    let (a, b) = span(c);
                    let mut best = [0.0f64; N_COMPONENTS];
                    for s in a..b {
                        let v = evidence::spectrum_components(
                            &ct,
                            &spec_tags[s],
                            &freq,
                            &pos[s],
                            want,
                            &params,
                        );
                        for k in 0..N_COMPONENTS {
                            best[k] = best[k].max(v[k]);
                        }
                    }
                    best
                })
                .collect();
            for (&i, v) in members[gi].iter().zip(got) {
                let row = &mut components[i as usize];
                for k in 0..N_COMPONENTS {
                    row[k] = row[k].max(v[k]);
                }
            }
        }

        // ---- fragment traces ----
        if p.trace.enabled {
            let w = p.trace.scans;
            let pools: Vec<traces::Pool> = (0..g.count.div_ceil(w))
                .into_par_iter()
                .map(|k| {
                    let part: Vec<(f64, Vec<(f64, f64)>)> = (k * w..((k + 1) * w).min(g.count))
                        .map(|t| {
                            let s = g.first + t;
                            let (a, b) = (sp.offsets[s], sp.offsets[s + 1]);
                            // Positive peaks with their intensities (kept for this component).
                            (
                                sp.rt[s],
                                (a..b)
                                    .filter(|&q| sp.positive[q])
                                    .map(|q| (sp.mz[q] as f64, sp.intensity[q] as f64))
                                    .collect(),
                            )
                        })
                        .collect();
                    traces::pool(&part, &p.trace)
                })
                .collect();
            let pool_rt: Vec<f64> = pools.iter().map(|q| q.mean_rt).collect();
            let coherence: Vec<HashMap<u32, f64>> = match p.trace.score {
                PrescreenTraceScore::TagCoherence => pools
                    .par_iter()
                    .map(|q| traces::pool_tag_coherence(q, alpha, tol, sigma, p.tags.max_charge))
                    .collect(),
                PrescreenTraceScore::CoherentFragments => Vec::new(),
            };
            let hist = Histogram::build(range.clone().map(|s| sp.peaks(s)), sp.nbin);
            let got: Vec<f64> = members[gi]
                .par_iter()
                .map_init(Vec::new, |frag, &i| {
                    let i = i as usize;
                    let Some(c) = cands[i].as_ref() else {
                        return 0.0;
                    };
                    let (a, b) = match c.rt {
                        Some((lo, hi)) => (
                            pool_rt.partition_point(|&r| r < lo),
                            pool_rt.partition_point(|&r| r <= hi),
                        ),
                        None => (0, pools.len()),
                    };
                    let mut best = 0.0f64;
                    match p.trace.score {
                        PrescreenTraceScore::TagCoherence => {
                            let Some(states) = view.states[i].as_ref() else {
                                return 0.0;
                            };
                            let mut by_code: Vec<Vec<u32>> = Vec::new();
                            let mut codes: Vec<u32> = Vec::new();
                            for w3 in states.windows(3) {
                                let (Some(x), Some(y), Some(z)) = (w3[0], w3[1], w3[2]) else {
                                    continue;
                                };
                                let code = alpha.canonical(x, y, z);
                                if codes.contains(&code) {
                                    continue;
                                }
                                codes.push(code);
                                by_code.push(
                                    (1..=p.tags.max_charge.min(c.nz))
                                        .map(|zz| alpha.key(code, zz, false))
                                        .collect(),
                                );
                            }
                            for pool in &coherence[a..b] {
                                best = best.max(traces::candidate_coherence(&by_code, pool));
                            }
                        }
                        PrescreenTraceScore::CoherentFragments => {
                            let cm = masses_of(i);
                            let sqrt_l = (cm.len() as f64).sqrt();
                            let weight = |m: f64| {
                                -hist.probability(m).ln()
                                    * if m < p.low_mz_threshold {
                                        p.low_mz_weight
                                    } else {
                                        1.0
                                    }
                            };
                            for rev in [false, true] {
                                fragment_mz(&cm, rev, c.nz, frag);
                                for pool in &pools[a..b] {
                                    best = best.max(traces::coherent_fragments(
                                        frag,
                                        pool,
                                        weight,
                                        p.frag_tol_da,
                                        p.trace.min_quality,
                                        sqrt_l,
                                    ));
                                }
                            }
                        }
                    }
                    best
                })
                .collect();
            for (&i, v) in members[gi].iter().zip(got) {
                let r = &mut components[i as usize][TRACE];
                *r = r.max(v);
            }
        }

        // ---- blind mass hypotheses ----
        if p.mass_hypotheses.enabled {
            let mh = &p.mass_hypotheses;
            let found: Vec<(usize, Vec<Hypothesis>, Vec<u8>)> = (0..g.count)
                .into_par_iter()
                .filter(|&t| sample[g.first + t])
                .map(|t| {
                    let hy = masses::infer_spectrum(&pos[t], g.lower, g.upper, alpha, tol, sigma);
                    let links: Vec<u8> = match (mh.ms1, ms1) {
                        (true, Some(m1)) => hy
                            .iter()
                            .map(|h| masses::ms1_link(h, rts[t], m1, mh.ms1_rt_s, mh.ms1_ppm))
                            .collect(),
                        _ => vec![0; hy.len()],
                    };
                    (t, hy, links)
                })
                .collect();
            let by_spectrum: HashMap<usize, Vec<Hypothesis>> = found
                .iter()
                .map(|(t, hy, links)| {
                    let kept: Vec<Hypothesis> = hy
                        .iter()
                        .zip(links)
                        .filter(|(_, &l)| !mh.require_ms1 || l > 0)
                        .map(|(h, _)| h.clone())
                        .collect();
                    (*t, kept)
                })
                .collect();
            for (t, hy, links) in found {
                for (h, l) in hy.into_iter().zip(links) {
                    hyps_out.push(((g.first + t) as u32, rts[t], h, l));
                }
            }
            if !by_spectrum.is_empty() {
                let got: Vec<f64> = members[gi]
                    .par_iter()
                    .map(|&i| {
                        let Some(c) = cands[i as usize].as_ref() else {
                            return 0.0;
                        };
                        let neutral: f64 = masses_of(i as usize).iter().sum::<f64>()
                            + mumdia_core::constants::WATER;
                        let (a, b) = span(c);
                        (a..b)
                            .filter_map(|t| by_spectrum.get(&t))
                            .map(|hy| masses::support(neutral, hy))
                            .fold(0.0, f64::max)
                    })
                    .collect();
                for (&i, v) in members[gi].iter().zip(got) {
                    let r = &mut components[i as usize][MASS];
                    *r = r.max(v);
                }
            }
        }
    }

    let mut stats = serde_json::Map::new();
    if need_spectrum_tags {
        stats.insert("tag_paths".into(), paths.load(Ordering::Relaxed).into());
        stats.insert(
            "tag_records".into(),
            tag_records.load(Ordering::Relaxed).into(),
        );
        stats.insert("tag_alphabet".into(), serde_json::json!(alpha.names));
        stats.insert("tag_unsupported_candidates".into(), view.unsupported.into());
    }
    if tag_retrieval {
        stats.insert("retrieval_index_keys".into(), index_keys.into());
        stats.insert("retrieval_forms_examined".into(), examined.into());
        stats.insert(
            "retrieval_forms_skipped_by_family".into(),
            skipped_by_family.into(),
        );
        stats.insert("families".into(), view.families.len().into());
    }
    if p.mass_hypotheses.enabled {
        stats.insert("mass_hypotheses".into(), hyps_out.len().into());
        stats.insert(
            "mass_hypothesis_spectra".into(),
            sample.iter().filter(|&&x| x).count().into(),
        );
    }
    Extended {
        retrieved: tag_retrieval.then_some(retrieved),
        components,
        stats,
        hypotheses: hyps_out,
    }
}
