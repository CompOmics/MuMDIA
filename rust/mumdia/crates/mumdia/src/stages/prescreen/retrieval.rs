//! Candidate retrieval from database-free tags, used by the tag prefilter.
//!
//! Library sequences enter here for the first time: tags were discovered from spectra alone
//! (`tags::discover`). A candidate is retrieved when one of its own trimers, in its own
//! residue states, at a fragment charge it can carry, was observed in an eligible spectrum of
//! its isolation window inside its RT bounds.
//!
//! Two implementations answer the same question and are tested for complete set agreement:
//! `eager` walks each candidate's eligible spectra; `ObservedIndex` inverts the window's tags
//! into key -> retention times once and answers each candidate by lookup.

use std::collections::HashMap;

use super::tags::{Alphabet, SpectrumTags};

/// A candidate's distinct tag keys at fragment charges `1..=zmax`. Trimers covering a position
/// the alphabet cannot express are skipped.
pub fn candidate_keys(states: &[Option<u16>], zmax: i32, alpha: &Alphabet) -> Vec<u32> {
    let mut v = Vec::new();
    for w in states.windows(3) {
        let (Some(a), Some(b), Some(c)) = (w[0], w[1], w[2]) else {
            continue;
        };
        let code = alpha.canonical(a, b, c);
        for z in 1..=zmax {
            v.push(alpha.key(code, z));
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Eager retrieval: any key in any spectrum of `tags` (already the eligible range).
pub fn eager(keys: &[u32], tags: &[SpectrumTags]) -> bool {
    tags.iter()
        .any(|t| keys.iter().any(|k| t.keys.binary_search(k).is_ok()))
}

/// One window's observed keys inverted into key -> ascending retention times.
pub struct ObservedIndex {
    map: HashMap<u32, Vec<f64>>,
}

impl ObservedIndex {
    /// `tags[i]` belongs to the spectrum at `rt[i]`, `rt` ascending.
    pub fn build(tags: &[SpectrumTags], rt: &[f64]) -> ObservedIndex {
        let mut map: HashMap<u32, Vec<f64>> = HashMap::new();
        for (t, &r) in tags.iter().zip(rt) {
            // `t.keys` is ascending without repeats.
            for &k in &t.keys {
                map.entry(k).or_default().push(r);
            }
        }
        ObservedIndex { map }
    }

    /// Any key observed at a retention time inside `rt` (inclusive; `None` = anywhere).
    pub fn any(&self, keys: &[u32], rt: Option<(f64, f64)>) -> bool {
        keys.iter().any(|k| {
            self.map.get(k).is_some_and(|r| match rt {
                None => !r.is_empty(),
                Some((lo, hi)) => {
                    let i = r.partition_point(|&x| x < lo);
                    i < r.len() && r[i] <= hi
                }
            })
        })
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tags::{discover, Alphabet};
    use super::*;

    fn alpha() -> Alphabet {
        Alphabet::build(
            &[(b'C', "Carbamidomethyl".into())],
            &[(b'M', "Oxidation".into())],
        )
        .unwrap()
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, m: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % m
        }
    }

    /// Random peptides, some oxidised, and random spectra with some planted b ladders.
    fn fixture(a: &Alphabet) -> (Vec<String>, Vec<SpectrumTags>, Vec<f64>) {
        let mut r = Lcg(42);
        let aa = b"ACDEFGHLMNPQSTVWYKR";
        let mut peps = Vec::new();
        for _ in 0..300 {
            let len = 7 + r.next(8) as usize;
            let mut s = String::new();
            for _ in 0..len {
                let c = aa[r.next(aa.len() as u64) as usize] as char;
                s.push(c);
                if c == 'C' {
                    s.push_str("[Carbamidomethyl]");
                }
                if c == 'M' && r.next(2) == 0 {
                    s.push_str("[Oxidation]");
                }
            }
            peps.push(s);
        }
        let mut tags = Vec::new();
        let mut rts = Vec::new();
        for k in 0..60 {
            let mut mz: Vec<f64> = (0..150)
                .map(|_| 150.0 + r.next(1_500_000) as f64 / 1000.0)
                .collect();
            let pf = &peps[(k * 5) % peps.len()];
            let st = a.tokenise(pf).unwrap();
            let mut m = 1.007_276;
            for s in st.iter().flatten() {
                m += a.masses[*s as usize];
                mz.push(m);
            }
            mz.sort_by(f64::total_cmp);
            tags.push(discover(&mz, a, 0.005, 2));
            rts.push(k as f64 * 10.0);
        }
        (peps, tags, rts)
    }

    /// Indexed retrieval returns exactly the eager candidate set.
    #[test]
    fn indexed_and_eager_retrieval_agree_completely() {
        let a = alpha();
        let (peps, tags, rts) = fixture(&a);
        let idx = ObservedIndex::build(&tags, &rts);
        let mut eager_set = Vec::new();
        let mut index_set = Vec::new();
        let windows: Vec<(f64, f64)> = (0..peps.len())
            .map(|i| ((i % 60) as f64 * 10.0 - 25.0, (i % 60) as f64 * 10.0 + 25.0))
            .collect();
        let states: Vec<Vec<Option<u16>>> = peps.iter().map(|p| a.tokenise(p).unwrap()).collect();
        for (i, s) in states.iter().enumerate() {
            let keys = candidate_keys(s, 2, &a);
            let (lo, hi) = windows[i];
            let a0 = rts.partition_point(|&r| r < lo);
            let b0 = rts.partition_point(|&r| r <= hi);
            if eager(&keys, &tags[a0..b0]) {
                eager_set.push(i);
            }
            if idx.any(&keys, Some((lo, hi))) {
                index_set.push(i);
            }
        }
        assert!(!eager_set.is_empty() && eager_set.len() < peps.len());
        assert_eq!(index_set, eager_set);
    }

    /// A modification outside the observed tag does not block retrieval: the backbone trimer
    /// in front of an oxidised M retrieves the oxidised form.
    #[test]
    fn modifications_outside_the_tag_are_allowed() {
        let a = alpha();
        let plain = a.tokenise("GASTEM[Oxidation]K").unwrap();
        let mut mz = vec![200.0];
        for s in plain[..3].iter().flatten() {
            let last = *mz.last().unwrap();
            mz.push(last + a.masses[*s as usize]);
        }
        let t = discover(&mz, &a, 0.005, 1);
        let keys = candidate_keys(&plain, 1, &a);
        assert!(eager(&keys, std::slice::from_ref(&t)));
    }
}
