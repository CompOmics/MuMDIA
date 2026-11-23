//! Tag retrieval for the tag prefilter, run window group by window group so only one group's
//! tags are in memory at a time.

use std::sync::atomic::{AtomicU64, Ordering};

use mumdia_core::config::Config;
use rayon::prelude::*;

use super::retrieval::{self, ObservedIndex};
use super::tags::{self, Alphabet, SpectrumTags};
use super::{Candidate, Spectra};

/// What tag retrieval produced.
pub struct Retrieved {
    /// Retrieval outcome per candidate.
    pub retrieved: Vec<bool>,
    pub stats: serde_json::Map<String, serde_json::Value>,
}

/// The tag view of the candidates. Residue states and tag keys are derived from the
/// peptidoform when a candidate is examined, not stored: on a 200M-candidate library the stored
/// per-candidate vectors were most of an 87 GB peak.
pub struct TagView<'a> {
    pforms: &'a [&'a str],
    charge: &'a [i32],
    alpha: &'a Alphabet,
    zcap: i32,
}

impl TagView<'_> {
    pub fn new<'a>(
        pforms: &'a [&'a str],
        charge: &'a [i32],
        alpha: &'a Alphabet,
        cfg: &Config,
    ) -> TagView<'a> {
        let p = &cfg.prescreen;
        TagView {
            pforms,
            charge,
            alpha,
            zcap: p.tags.max_charge.min(p.max_frag_charge),
        }
    }

    pub fn keys(&self, i: usize) -> Vec<u32> {
        let zmax = self.zcap.min(self.charge[i]).max(1);
        match self.alpha.tokenise(self.pforms[i]) {
            Some(s) => retrieval::candidate_keys(&s, zmax, self.alpha),
            None => Vec::new(),
        }
    }
}

/// Retrieve the candidates one of whose trimers, in its own residue states, was observed in an
/// eligible spectrum of its isolation window. A candidate without an expressible trimer is
/// retrieved, because tags cannot judge it.
pub fn run(
    sp: &Spectra,
    cands: &[Option<Candidate>],
    members: &[Vec<u32>],
    view: &TagView<'_>,
    alpha: &Alphabet,
    cfg: &Config,
) -> Retrieved {
    let p = &cfg.prescreen;
    let mut retrieved = vec![false; cands.len()];
    let paths = AtomicU64::new(0);
    let tag_records = AtomicU64::new(0);
    let mut examined = 0u64;
    let mut index_keys = 0u64;
    let mut unsupported = 0u64;

    for (gi, g) in sp.groups.iter().enumerate() {
        if members[gi].is_empty() {
            continue;
        }
        let range = g.first..g.first + g.count;
        let rts = &sp.rt[range.clone()];
        let spec_tags: Vec<SpectrumTags> = range
            .into_par_iter()
            .map(|s| {
                let (a, b) = (sp.offsets[s], sp.offsets[s + 1]);
                let mz: Vec<f64> = (a..b)
                    .filter(|&k| sp.positive[k])
                    .map(|k| sp.mz[k] as f64)
                    .collect();
                let t = tags::discover(&mz, alpha, p.tags.edge_tol_da, p.tags.max_charge);
                paths.fetch_add(t.paths, Ordering::Relaxed);
                tag_records.fetch_add(t.keys.len() as u64, Ordering::Relaxed);
                t
            })
            .collect();
        let obs = ObservedIndex::build(&spec_tags, rts);
        index_keys += obs.len() as u64;
        // Per member: (retrieved, examined, not expressible).
        let got: Vec<(bool, bool, bool)> = members[gi]
            .par_iter()
            .map(|&i| {
                let i = i as usize;
                let Some(c) = cands[i].as_ref() else {
                    return (false, false, false);
                };
                let keys = view.keys(i);
                if keys.is_empty() {
                    return (true, false, true);
                }
                (obs.any(&keys, c.rt), true, false)
            })
            .collect();
        for (&i, (r, e, un)) in members[gi].iter().zip(got) {
            retrieved[i as usize] |= r;
            examined += e as u64;
            unsupported += un as u64;
        }
    }

    let mut stats = serde_json::Map::new();
    stats.insert("tag_paths".into(), paths.load(Ordering::Relaxed).into());
    stats.insert(
        "tag_records".into(),
        tag_records.load(Ordering::Relaxed).into(),
    );
    stats.insert("tag_alphabet".into(), serde_json::json!(alpha.names));
    stats.insert("tag_unsupported_candidates".into(), unsupported.into());
    stats.insert("retrieval_index_keys".into(), index_keys.into());
    stats.insert("retrieval_forms_examined".into(), examined.into());
    Retrieved { retrieved, stats }
}
