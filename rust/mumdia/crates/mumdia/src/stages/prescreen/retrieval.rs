//! Candidate retrieval from database-free tags, the backbone/position index, and delayed
//! modification-form enumeration.
//!
//! Library sequences enter here for the first time: tags were discovered from spectra alone
//! (`tags::discover`). A candidate is retrieved when one of its own trimers, in its own
//! residue states, at a fragment charge it can carry, was observed in an eligible spectrum of
//! its isolation window inside its RT bounds. Retrieval is permissive by design; a candidate
//! without such evidence takes the rescue route (scored anyway) unless `rescue` is off, and the
//! report counts the two separately, so retrieval losses and scoring losses are never mixed.
//!
//! Two implementations answer the same question and are tested for complete set agreement:
//! `eager` walks each candidate's eligible spectra; `ObservedIndex` inverts the window's tags
//! into key -> retention times once and answers each candidate by lookup.
//!
//! Delayed enumeration first retrieves backbone families (all forms sharing one residue
//! sequence) on the backbone states of their trimers, then examines forms only inside the
//! retrieved families. The per-form rule is the eager one, so the retrieved set is identical;
//! what is saved is the examination (or, in a FASTA build, the generation) of forms of families
//! with no supporting tag, which `forms_with_tag` makes concrete.

use std::collections::HashMap;

use super::tags::{Alphabet, SpectrumTags};

/// A candidate's distinct tag keys at fragment charges `1..=zmax`, both ladder kinds when
/// `gapped`. Trimers covering a position the alphabet cannot express are skipped.
pub fn candidate_keys(
    states: &[Option<u16>],
    zmax: i32,
    gapped: bool,
    alpha: &Alphabet,
) -> Vec<u32> {
    let mut v = Vec::new();
    for w in states.windows(3) {
        let (Some(a), Some(b), Some(c)) = (w[0], w[1], w[2]) else {
            continue;
        };
        let code = alpha.canonical(a, b, c);
        for z in 1..=zmax {
            v.push(alpha.key(code, z, false));
            if gapped {
                v.push(alpha.key(code, z, true));
            }
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// The same keys on backbone states (every variable modification dropped).
pub fn backbone_keys(
    states: &[Option<u16>],
    zmax: i32,
    gapped: bool,
    alpha: &Alphabet,
) -> Vec<u32> {
    let norm: Vec<Option<u16>> = states
        .iter()
        .map(|s| s.map(|x| alpha.normal[x as usize]))
        .collect();
    candidate_keys(&norm, zmax, gapped, alpha)
}

/// Re-key an observed key on backbone states.
pub fn normalise_key(key: u32, alpha: &Alphabet) -> u32 {
    let (code, z, gapped) = alpha.unkey(key);
    alpha.key(alpha.backbone(code), z, gapped)
}

/// A candidate's positioned keys: for every trimer whose ladder is a run of four of its own
/// fragments, b-ladders starting at b1..b(L-4) and y-ladders starting at y1..y(L-4), at fragment
/// charges `1..=zmax`, the start bin and its two neighbours (the observed start carries the
/// peak's mass error). With `both_orientations` the fully reversed residue array adds its own
/// keys, so a reversed decoy carries exactly its target's keys, as the fragment score does.
#[allow(clippy::needless_range_loop)]
pub fn positioned_keys(
    states: &[Option<u16>],
    zmax: i32,
    both_orientations: bool,
    alpha: &Alphabet,
) -> Vec<u64> {
    positioned_keys_n(states, zmax, both_orientations, alpha, 3)
}

/// `positioned_keys` for ladders of `residues` steps: 3 (four peaks) or 2 (three peaks).
#[allow(clippy::needless_range_loop)]
pub fn positioned_keys_n(
    states: &[Option<u16>],
    zmax: i32,
    both_orientations: bool,
    alpha: &Alphabet,
    residues: usize,
) -> Vec<u64> {
    use super::tags::{positioned_key, positioned_key2, start_bin};
    let w = residues;
    use mumdia_core::constants::WATER;
    let l = states.len();
    let mut out = Vec::new();
    if l < w + 2 {
        return out;
    }
    let orientations: &[bool] = if both_orientations {
        &[false, true]
    } else {
        &[false]
    };
    for &rev in orientations {
        let st = |i: usize| states[if rev { l - 1 - i } else { i }];
        let m = |i: usize| st(i).map(|x| alpha.masses[x as usize]);
        // Prefix sums; None once a position cannot be expressed.
        let mut prefix = vec![Some(0.0f64); l + 1];
        for i in 0..l {
            prefix[i + 1] = prefix[i].zip(m(i)).map(|(p, x)| p + x);
        }
        let mut suffix = vec![Some(WATER); l + 1];
        for r in 0..l {
            suffix[r + 1] = suffix[r].zip(m(l - 1 - r)).map(|(p, x)| p + x);
        }
        // A ladder of `w` residue steps starting at b_q (prefix of q residues) or at y_r
        // (suffix of r residues plus water); its end fragment must still be a fragment.
        let key = |s: &[Option<u16>], z: i32, bin: u64| -> Option<u64> {
            if w == 2 {
                Some(positioned_key2(alpha, [s[0]?, s[1]?], z, bin))
            } else {
                Some(positioned_key(alpha, [s[0]?, s[1]?, s[2]?], z, bin))
            }
        };
        for q in 1..=l - w - 1 {
            let b_steps: Vec<Option<u16>> = (0..w).map(|k| st(q + k)).collect();
            let y_steps: Vec<Option<u16>> = (0..w).map(|k| st(l - 1 - q - k)).collect();
            for (start, steps) in [(prefix[q], &b_steps), (suffix[q], &y_steps)] {
                let Some(start) = start else { continue };
                let bin = start_bin(start);
                for z in 1..=zmax {
                    for bb in bin.saturating_sub(1)..=bin + 1 {
                        out.extend(key(steps, z, bb));
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// One window's positioned tags inverted into key -> ascending retention times.
pub struct PositionedIndex {
    map: HashMap<u64, Vec<f64>>,
}

impl PositionedIndex {
    pub fn build(tags: &[SpectrumTags], rt: &[f64]) -> PositionedIndex {
        let mut map: HashMap<u64, Vec<f64>> = HashMap::new();
        for (t, &r) in tags.iter().zip(rt) {
            for &k in &t.positioned {
                map.entry(k).or_default().push(r);
            }
        }
        PositionedIndex { map }
    }

    /// Any key observed at a retention time inside `rt` (inclusive; `None` = anywhere).
    pub fn any(&self, keys: &[u64], rt: Option<(f64, f64)>) -> bool {
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
    /// `tags[i]` belongs to the spectrum at `rt[i]`, `rt` ascending. With `normalise`, keys
    /// are re-keyed on backbone states (the delayed family index).
    pub fn build(tags: &[SpectrumTags], rt: &[f64], normalise: Option<&Alphabet>) -> ObservedIndex {
        let mut map: HashMap<u32, Vec<f64>> = HashMap::new();
        for (t, &r) in tags.iter().zip(rt) {
            let mut keys: Vec<u32> = match normalise {
                Some(a) => t.keys.iter().map(|&k| normalise_key(k, a)).collect(),
                None => t.keys.clone(),
            };
            keys.sort_unstable();
            keys.dedup();
            for k in keys {
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

/// Library backbone/position postings: backbone trimer code -> `(family, position)`, plus the
/// number of families containing each code (the FASTA-side frequency, kept separate from the
/// spectrum-side one).
pub struct BackboneIndex {
    offsets: Vec<usize>,
    codes: Vec<u32>,
    family: Vec<u32>,
    position: Vec<u16>,
    pub n_families: usize,
}

impl BackboneIndex {
    /// `families[f]` is family `f`'s backbone states.
    pub fn build(families: &[Vec<u16>], alpha: &Alphabet) -> BackboneIndex {
        let mut occ: Vec<(u32, u32, u16)> = Vec::new();
        for (f, s) in families.iter().enumerate() {
            for p in 0..s.len().saturating_sub(2) {
                occ.push((
                    alpha.canonical(s[p], s[p + 1], s[p + 2]),
                    f as u32,
                    p as u16,
                ));
            }
        }
        occ.sort_unstable();
        let mut codes = Vec::new();
        let mut offsets = Vec::new();
        for (i, o) in occ.iter().enumerate() {
            if codes.last() != Some(&o.0) {
                codes.push(o.0);
                offsets.push(i);
            }
        }
        offsets.push(occ.len());
        BackboneIndex {
            offsets,
            codes,
            family: occ.iter().map(|o| o.1).collect(),
            position: occ.iter().map(|o| o.2).collect(),
            n_families: families.len(),
        }
    }

    /// Postings of a backbone code: `(family, position)` ascending.
    pub fn postings(&self, code: u32) -> Vec<(u32, u16)> {
        match self.codes.binary_search(&code) {
            Ok(i) => (self.offsets[i]..self.offsets[i + 1])
                .map(|k| (self.family[k], self.position[k]))
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Families containing `code` at least once.
    pub fn document_frequency(&self, code: u32) -> usize {
        let mut f: Vec<u32> = self.postings(code).into_iter().map(|p| p.0).collect();
        f.dedup();
        f.len()
    }

    /// FASTA-side information `-ln((docs + 0.5) / (families + 1))`.
    pub fn information(&self, code: u32) -> f64 {
        -((self.document_frequency(code) as f64 + 0.5) / (self.n_families as f64 + 1.0)).ln()
    }
}

/// Linear-scan postings, the reference the index is tested against.
pub fn linear_postings(families: &[Vec<u16>], code: u32, alpha: &Alphabet) -> Vec<(u32, u16)> {
    let mut v = Vec::new();
    for (f, s) in families.iter().enumerate() {
        for p in 0..s.len().saturating_sub(2) {
            if alpha.canonical(s[p], s[p + 1], s[p + 2]) == code {
                v.push((f as u32, p as u16));
            }
        }
    }
    v
}

/// A variable modification for form enumeration: a backbone state and its modified state.
#[derive(Clone, Copy, Debug)]
pub struct VarMod {
    pub plain: u16,
    pub modified: u16,
}

/// Enumeration-first: every form of `backbone` with at most `max_var` variable modifications
/// (one per position), as state vectors, sorted.
pub fn enumerate_forms(backbone: &[u16], var: &[VarMod], max_var: usize) -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    let mut cur = backbone.to_vec();
    fn rec(p: usize, left: usize, cur: &mut Vec<u16>, var: &[VarMod], out: &mut Vec<Vec<u16>>) {
        if p == cur.len() {
            out.push(cur.clone());
            return;
        }
        rec(p + 1, left, cur, var, out);
        if left > 0 {
            let plain = cur[p];
            for v in var.iter().filter(|v| v.plain == plain) {
                cur[p] = v.modified;
                rec(p + 1, left - 1, cur, var, out);
                cur[p] = plain;
            }
        }
    }
    rec(0, max_var, &mut cur, var, &mut out);
    out.sort();
    out
}

/// Delayed: only the forms whose states at `position..position + 3` equal `tag` (in sequence
/// order), modifications outside the tag left free. The prototype's `oxidation_forms`
/// generalised to any number of variable modification types. Equal, as a set, to
/// `enumerate_forms` filtered on the tag (tested).
pub fn forms_with_tag(
    backbone: &[u16],
    var: &[VarMod],
    max_var: usize,
    position: usize,
    tag: [u16; 3],
) -> Vec<Vec<u16>> {
    if position + 3 > backbone.len() {
        return Vec::new();
    }
    let mut fixed = backbone.to_vec();
    let mut used = 0usize;
    for (k, &t) in tag.iter().enumerate() {
        let p = position + k;
        if t == backbone[p] {
            continue;
        }
        if var
            .iter()
            .any(|v| v.plain == backbone[p] && v.modified == t)
        {
            fixed[p] = t;
            used += 1;
        } else {
            return Vec::new();
        }
    }
    if used > max_var {
        return Vec::new();
    }
    let mut out = Vec::new();
    fn rec(
        p: usize,
        left: usize,
        cur: &mut Vec<u16>,
        var: &[VarMod],
        lock: std::ops::Range<usize>,
        out: &mut Vec<Vec<u16>>,
    ) {
        if p == cur.len() {
            out.push(cur.clone());
            return;
        }
        rec(p + 1, left, cur, var, lock.clone(), out);
        if left > 0 && !lock.contains(&p) {
            let plain = cur[p];
            for v in var.iter().filter(|v| v.plain == plain) {
                cur[p] = v.modified;
                rec(p + 1, left - 1, cur, var, lock.clone(), out);
                cur[p] = plain;
            }
        }
    }
    rec(
        0,
        max_var - used,
        &mut fixed,
        var,
        position..position + 3,
        &mut out,
    );
    out.sort();
    out
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
            tags.push(discover(&mz, a, 0.005, 2, None, false));
            rts.push(k as f64 * 10.0);
        }
        (peps, tags, rts)
    }

    /// Indexed retrieval returns exactly the eager candidate set, and the delayed (family
    /// first) retrieval returns the same set while examining fewer forms.
    #[test]
    fn indexed_eager_and_delayed_retrieval_agree_completely() {
        let a = alpha();
        let (peps, tags, rts) = fixture(&a);
        let idx = ObservedIndex::build(&tags, &rts, None);
        let norm = ObservedIndex::build(&tags, &rts, Some(&a));
        let mut eager_set = Vec::new();
        let mut index_set = Vec::new();
        let mut delayed_set = Vec::new();
        let mut examined = 0;
        let windows: Vec<(f64, f64)> = (0..peps.len())
            .map(|i| ((i % 60) as f64 * 10.0 - 25.0, (i % 60) as f64 * 10.0 + 25.0))
            .collect();
        // Families: same backbone states.
        let states: Vec<Vec<Option<u16>>> = peps.iter().map(|p| a.tokenise(p).unwrap()).collect();
        let mut fam_hit = std::collections::HashSet::new();
        for (i, s) in states.iter().enumerate() {
            let (lo, hi) = windows[i];
            if norm.any(&backbone_keys(s, 2, false, &a), Some((lo, hi))) {
                let b: Vec<u16> = s.iter().map(|x| a.normal[x.unwrap() as usize]).collect();
                fam_hit.insert(b);
            }
        }
        for (i, s) in states.iter().enumerate() {
            let keys = candidate_keys(s, 2, false, &a);
            let (lo, hi) = windows[i];
            let a0 = rts.partition_point(|&r| r < lo);
            let b0 = rts.partition_point(|&r| r <= hi);
            if eager(&keys, &tags[a0..b0]) {
                eager_set.push(i);
            }
            if idx.any(&keys, Some((lo, hi))) {
                index_set.push(i);
            }
            let b: Vec<u16> = s.iter().map(|x| a.normal[x.unwrap() as usize]).collect();
            if fam_hit.contains(&b) {
                examined += 1;
                if idx.any(&keys, Some((lo, hi))) {
                    delayed_set.push(i);
                }
            }
        }
        assert!(!eager_set.is_empty() && eager_set.len() < peps.len());
        assert_eq!(index_set, eager_set);
        assert_eq!(delayed_set, eager_set);
        assert!(
            examined < peps.len(),
            "families without a tag are not examined"
        );
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
        let t = discover(&mz, &a, 0.005, 1, None, false);
        let keys = candidate_keys(&plain, 1, false, &a);
        assert!(eager(&keys, std::slice::from_ref(&t)));
    }

    #[test]
    fn the_backbone_index_returns_the_linear_scan_postings() {
        let a = alpha();
        let (peps, _, _) = fixture(&a);
        let fams: Vec<Vec<u16>> = peps
            .iter()
            .map(|p| {
                a.tokenise(p)
                    .unwrap()
                    .iter()
                    .map(|x| a.normal[x.unwrap() as usize])
                    .collect()
            })
            .collect();
        let ix = BackboneIndex::build(&fams, &a);
        let mut checked = 0;
        for f in fams.iter().take(40) {
            for p in 0..f.len() - 2 {
                let code = a.canonical(f[p], f[p + 1], f[p + 2]);
                assert_eq!(ix.postings(code), linear_postings(&fams, code, &a));
                checked += 1;
            }
        }
        assert!(checked > 100);
        assert_eq!(ix.postings(u32::MAX - 1), Vec::new());
    }

    /// Delayed generation equals enumeration-first generation filtered on the tag, for every
    /// tag position and state combination, with two variable modification types and both one
    /// and three allowed modifications.
    #[test]
    fn delayed_form_generation_matches_enumeration_first() {
        let a = Alphabet::build(
            &[(b'C', "Carbamidomethyl".into())],
            &[(b'M', "Oxidation".into()), (b'S', "Phospho".into())],
        )
        .unwrap();
        let st = |r: u8, d: f64| a.state(r, d).unwrap();
        let var = [
            VarMod {
                plain: st(b'M', 0.0),
                modified: st(b'M', 15.994_914_62),
            },
            VarMod {
                plain: st(b'S', 0.0),
                modified: st(b'S', 79.966_331_09),
            },
        ];
        let backbone: Vec<u16> = a
            .tokenise("MSAMSEMK")
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap())
            .collect();
        let mut cases = 0;
        for max_var in [1, 3] {
            let all = enumerate_forms(&backbone, &var, max_var);
            for pos in 0..backbone.len() - 2 {
                let mut tags: Vec<[u16; 3]> = all
                    .iter()
                    .map(|f| [f[pos], f[pos + 1], f[pos + 2]])
                    .collect();
                tags.sort();
                tags.dedup();
                for tag in tags {
                    let want: Vec<Vec<u16>> = all
                        .iter()
                        .filter(|f| f[pos..pos + 3] == tag)
                        .cloned()
                        .collect();
                    assert_eq!(forms_with_tag(&backbone, &var, max_var, pos, tag), want);
                    cases += 1;
                }
            }
        }
        assert!(cases > 30, "{cases}");
        // A tag state the chemistry does not allow yields nothing.
        let bogus = [st(b'A', 0.0), st(b'A', 0.0), st(b'A', 0.0)];
        assert!(forms_with_tag(&backbone, &var, 3, 0, bogus).is_empty());
    }

    /// Positioned keys: a b-ladder planted at the candidate's own prefix mass is found; the same
    /// trimer at a different start mass is not; a full reversal carries the same keys with both
    /// orientations.
    #[test]
    fn positioned_tags_require_the_candidates_own_ladder_position() {
        let a = alpha();
        let st = a.tokenise("GASPVTLEK").unwrap();
        let m = |i: usize| a.masses[st[i].unwrap() as usize];
        // b2..b5 ladder: prefix(2) then S, P, V.
        let start = m(0) + m(1);
        let mz: Vec<f64> = (0..4)
            .map(|k| start + (2..2 + k).map(m).sum::<f64>() + mumdia_core::constants::PROTON)
            .collect();
        let t = discover(&mz, &a, 0.005, 1, None, true);
        let keys = positioned_keys(&st, 1, false, &a);
        assert!(t.positioned.iter().any(|k| keys.binary_search(k).is_ok()));
        let shifted: Vec<f64> = mz.iter().map(|x| x + 0.5).collect();
        let t2 = discover(&shifted, &a, 0.005, 1, None, true);
        assert!(!t2.keys.is_empty(), "the trimer itself is still there");
        assert!(!t2.positioned.iter().any(|k| keys.binary_search(k).is_ok()));
        let rev = a.tokenise("KELTVPSAG").unwrap();
        assert_eq!(
            positioned_keys(&st, 2, true, &a),
            positioned_keys(&rev, 2, true, &a)
        );
        assert!(positioned_keys(&a.tokenise("GASP").unwrap(), 2, true, &a).is_empty());
        // Two-residue ladders: three peaks b2..b4 suffice.
        let three: Vec<f64> = mz[..3].to_vec();
        let t3 = discover(&three, &a, 0.005, 1, None, true);
        let k2 = positioned_keys_n(&st, 1, false, &a, 2);
        assert!(t3.positioned.iter().any(|k| k2.binary_search(k).is_ok()));
        assert!(!t3.positioned.iter().any(|k| keys.binary_search(k).is_ok()));
    }
}
