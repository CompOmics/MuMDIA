//! Sequence-tag prescan `mumdia prescan`: keep only the modification-bearing library candidates
//! whose modification-anchored sequence trimers are actually observed in their own isolation
//! window and retention-time range.
//!
//! Purpose. A modification search multiplies the candidate space by the number of modforms per
//! peptide, and most of those hypotheses have no supporting signal in a given run. This stage
//! prunes them per file BEFORE the library is assembled, so extraction and rescoring see a search
//! space sized to the evidence rather than to the enumeration.
//!
//! What it is NOT. The screen cannot discriminate a true modified peptide from its decoy, and it is
//! not meant to. `anchored_tris` emits every trimer in both orientations, and a reverse decoy
//! preserves both composition and precursor m/z, so a decoy's anchored tag set is identical to its
//! target's: a decoy survives exactly when its target does. Measured target:decoy survival ratio is
//! 1.00 to two decimals on every run tested. That is the property that makes this safe, because it
//! leaves target/decoy exchangeability untouched and therefore leaves downstream FDR valid. Treat
//! it as a compute reduction, never as a discriminator.
//!
//! Both labels are screened by the identical criterion, each on its own sequence, m/z and RT
//! window. Screening only targets and then admitting their paired decoys makes the surviving
//! targets signal-enriched while their decoys are signal-blind, which biases the modification's
//! q-values anticonservatively.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::PrescanConfig;
use mumdia_core::constants::residue_mass;
use mumdia_core::mass::unimod_mass;
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, TableFile};
use rayon::prelude::*;
use serde_json::json;
use tracing::info;

pub struct PrescanParams<'a> {
    /// `spectra_ms2` for this run.
    pub ms2: &'a str,
    /// `isolation_windows` for this run.
    pub isolation_windows: &'a str,
    /// Library precursors: `candidate_id, peptidoform, precursor_mz, label`.
    pub library_precursors: &'a str,
    /// Per-candidate RT bounds (`candidate_id, rt_lo, rt_hi`), i.e. a `run_windows`-shaped table.
    pub run_windows: &'a str,
    pub out: &'a str,
    pub cfg: &'a PrescanConfig,
    pub config_hash: &'a str,
}

/// Residue alphabet for tags: the 20 standard residues with I/L merged (isobaric, so a mass delta
/// cannot tell them apart), plus one entry per configured modified residue.
///
/// Masses come from the shared model, never a local copy: `residue_mass` for the backbone and
/// `unimod_mass` for the modification delta. A modified residue's tag mass is residue + delta.
struct Alphabet {
    masses: Vec<f64>,
    /// (residue, mod name) -> alphabet index, for the modified entries.
    modded: HashMap<(u8, String), usize>,
    /// residue -> alphabet index, for the unmodified entries.
    plain: HashMap<u8, usize>,
    /// Alphabet indices that count as "the modification of interest" for anchoring.
    anchor: HashSet<usize>,
}

impl Alphabet {
    fn build(cfg: &PrescanConfig) -> Result<Self> {
        // I and L share an index: a residue-mass delta cannot distinguish them, so treating them
        // as distinct would invent tags the spectrum can never confirm or refute.
        const PLAIN: &[u8] = b"GASPVTCLNDQKEMHFRYW";
        let mut masses = Vec::new();
        let mut plain = HashMap::new();
        for &aa in PLAIN {
            let m = residue_mass(aa)
                .ok_or_else(|| anyhow::anyhow!("no residue mass for '{}'", aa as char))?;
            plain.insert(aa, masses.len());
            if aa == b'L' {
                plain.insert(b'I', masses.len());
            }
            masses.push(m);
        }
        let mut modded = HashMap::new();
        let mut anchor = HashSet::new();
        for spec in cfg.mods.iter().chain(cfg.anchor_mods.iter()) {
            let (residue, name) = parse_mod_spec(spec)?;
            let base = residue_mass(residue)
                .ok_or_else(|| anyhow::anyhow!("no residue mass for '{}'", residue as char))?;
            let delta = unimod_mass(&name).ok_or_else(|| {
                anyhow::anyhow!(
                    "prescan.mods/anchor_mods names an unknown modification '{name}'; \
                     add it to the shared mass model rather than hard-coding a delta here"
                )
            })?;
            let key = (residue, name);
            if let std::collections::hash_map::Entry::Vacant(e) = modded.entry(key.clone()) {
                e.insert(masses.len());
                masses.push(base + delta);
            }
        }
        // Anchors are resolved after every mod has an index, so an anchor also listed in `mods`
        // reuses the same entry instead of duplicating it.
        for spec in &cfg.anchor_mods {
            let (residue, name) = parse_mod_spec(spec)?;
            if let Some(&i) = modded.get(&(residue, name)) {
                anchor.insert(i);
            }
        }
        Ok(Alphabet {
            masses,
            modded,
            plain,
            anchor,
        })
    }

    /// Tokenise a ProForma-lite peptidoform into alphabet indices. `None` when the sequence
    /// contains a residue or modification outside the alphabet: such a candidate cannot be tagged
    /// meaningfully, so it is dropped rather than screened on a partial sequence.
    fn tokenise(&self, pform: &str) -> Option<Vec<usize>> {
        let b = pform.as_bytes();
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            let aa = b[i];
            if !aa.is_ascii_alphabetic() {
                return None;
            }
            i += 1;
            if i < b.len() && b[i] == b'[' {
                let end = pform[i..].find(']')? + i;
                let name = &pform[i + 1..end];
                out.push(*self.modded.get(&(aa, name.to_string()))?);
                i = end + 1;
            } else {
                out.push(*self.plain.get(&aa)?);
            }
        }
        Some(out)
    }

    /// Trimers covering an anchored (modified) position, in both b and y orientation.
    ///
    /// Anchoring is the point: a trimer elsewhere in the peptide is evidence for the backbone, not
    /// for the modification, and would let unmodified signal keep a modified hypothesis alive.
    fn anchored_tris(&self, idx: &[usize]) -> HashSet<Tri> {
        let mut tris = HashSet::new();
        let l = idx.len();
        for p in 0..l {
            if !self.anchor.contains(&idx[p]) {
                continue;
            }
            for a in p.saturating_sub(2)..=p {
                if a + 2 < l {
                    let t = (idx[a] as u32, idx[a + 1] as u32, idx[a + 2] as u32);
                    tris.insert(t);
                    tris.insert((t.2, t.1, t.0));
                }
            }
        }
        tris
    }
}

fn parse_mod_spec(spec: &str) -> Result<(u8, String)> {
    // "C:Farnesyl" -> (b'C', "Farnesyl")
    let (r, name) = spec.split_once(':').ok_or_else(|| {
        anyhow::anyhow!("prescan mod spec '{spec}' must be RESIDUE:ModName, e.g. C:Farnesyl")
    })?;
    let rb = r.as_bytes();
    if rb.len() != 1 || !rb[0].is_ascii_alphabetic() {
        anyhow::bail!("prescan mod spec '{spec}' has a non-single-residue target '{r}'");
    }
    Ok((rb[0].to_ascii_uppercase(), name.to_string()))
}

/// One trimer tag: three alphabet indices in sequence order.
type Tri = (u32, u32, u32);
/// Index cell key: (isolation window id, RT bin).
type Cell = (u32, i64);
/// Observed trimer tags per (isolation window, RT bin).
type ObsIndex = HashMap<Cell, HashSet<Tri>>;

/// Chained charge-1 residue-mass deltas within one spectrum -> observed trimer tags.
///
/// Deliberately permissive: raw peak deltas, no deisotoping and no charge deconvolution. A tag is
/// a cheap necessary condition, and a false tag only fails to prune, whereas a missed tag discards
/// a real candidate irrecoverably.
fn spectrum_trimers(mz: &[f64], alpha: &Alphabet, tol: f64) -> HashSet<Tri> {
    let n = mz.len();
    let mut succ: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
    for (r, &rm) in alpha.masses.iter().enumerate() {
        let mut j0 = 0usize;
        for i in 0..n {
            let lo = mz[i] + rm - tol;
            let hi = mz[i] + rm + tol;
            // mz is ascending, and lo grows with i, so the scan start only moves forward.
            if j0 < i + 1 {
                j0 = i + 1;
            }
            while j0 < n && mz[j0] < lo {
                j0 += 1;
            }
            let mut j = j0;
            while j < n && mz[j] <= hi {
                succ[i].push((j, r as u32));
                j += 1;
            }
        }
    }
    let mut tri = HashSet::new();
    for i in 0..n {
        for &(j, r1) in &succ[i] {
            for &(k, r2) in &succ[j] {
                for &(_, r3) in &succ[k] {
                    tri.insert((r1, r2, r3));
                }
            }
        }
    }
    tri
}

pub fn run(p: PrescanParams) -> Result<u64> {
    let t0 = Instant::now();
    // `--out` must not be one of the inputs. The library is fully read before the
    // survivors table is written, so writing over it replaced a complete precursor
    // library with a two-column list and exited 0.
    mumdia_io::refuse_output_over_input(
        p.out,
        &[
            ("--lib-precursors", p.library_precursors),
            ("--run-windows", p.run_windows),
            ("--ms2", p.ms2),
            ("--isolation-windows", p.isolation_windows),
        ],
    )?;
    let alpha = Alphabet::build(p.cfg)?;
    if alpha.anchor.is_empty() {
        anyhow::bail!(
            "prescan.anchor_mods is empty, so no candidate can ever be anchored and every \
             modified candidate would be discarded; list the modifications to screen for"
        );
    }

    // ---- observed tag index, parallel over spectra ----
    let ms2 = TableFile::open(p.ms2)?;
    let wid = ms2.u32("window_id")?;
    let rts = ms2.f64("rt_seconds")?;
    let mzs = ms2.list_f32("mz")?;
    let ints = ms2.list_f32("intensity")?;
    let bin = p.cfg.rt_bin_s;
    let top = p.cfg.top_peaks;
    let tol = p.cfg.tol_da;

    let per_spectrum: Vec<(Cell, HashSet<Tri>)> = (0..ms2.nrows)
        .into_par_iter()
        .filter_map(|s| {
            let raw = &mzs[s];
            if raw.len() < 4 {
                return None;
            }
            // Keep the `top` most intense peaks, then sort by m/z: the delta scan needs ascending
            // m/z, and the intensity cut bounds the O(peaks^2) edge search.
            let mut peaks: Vec<(f64, f32)> = raw
                .iter()
                .zip(ints[s].iter())
                .map(|(&m, &i)| (m as f64, i))
                .collect();
            if top > 0 && peaks.len() > top {
                // Tie-break by ascending m/z, not by whatever the sort happens to do. The cut is
                // deep in the peak distribution (median ~1300 peaks per spectrum here), so the
                // 150th and 151st peaks are often near-equal in intensity and the choice between
                // them is arbitrary. Measured: moving the cut by ONE peak changes the survivor set
                // by 3.2%, so an unspecified tie order would make reruns differ by more than most
                // parameter changes. Deterministic ordering is required wherever floats are
                // reduced; this is the same rule applied to peak selection.
                peaks.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.total_cmp(&b.0)));
                peaks.truncate(top);
            }
            peaks.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let mz: Vec<f64> = peaks.into_iter().map(|(m, _)| m).collect();
            let tri = spectrum_trimers(&mz, &alpha, tol);
            if tri.is_empty() {
                return None;
            }
            Some(((wid[s], (rts[s] / bin).floor() as i64), tri))
        })
        .collect();
    let mut obs: ObsIndex = HashMap::new();
    for (k, v) in per_spectrum {
        obs.entry(k).or_default().extend(v);
    }
    info!(
        cells = obs.len(),
        spectra = ms2.nrows,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "prescan: built observed tag index"
    );
    drop(ms2);

    // ---- isolation windows ----
    let win = TableFile::open(p.isolation_windows)?;
    let w_id = win.u32("window_id")?;
    let w_lo = win.f64("lower")?;
    let w_hi = win.f64("upper")?;

    // ---- library + per-candidate RT bounds ----
    let lib = TableFile::open(p.library_precursors)?;
    let cid = lib.u32("candidate_id")?;
    let pform = lib.str("peptidoform")?;
    let pmz = lib.f64("precursor_mz")?;
    let label = lib.str("label")?;
    let rw = TableFile::open(p.run_windows)?;
    let r_cid = rw.u32("candidate_id")?;
    let r_lo = rw.f64("rt_lo")?;
    let r_hi = rw.f64("rt_hi")?;
    // Join by candidate_id through a dense lookup: both tables are row-aligned in practice, but
    // relying on that silently mismatches RT bounds if either is ever reordered.
    let maxc = cid.iter().copied().max().unwrap_or(0) as usize;
    let mut lo_by = vec![f64::NAN; maxc + 1];
    let mut hi_by = vec![f64::NAN; maxc + 1];
    for i in 0..rw.nrows {
        let c = r_cid[i] as usize;
        if c <= maxc {
            lo_by[c] = r_lo[i];
            hi_by[c] = r_hi[i];
        }
    }
    drop(rw);

    let slack = p.cfg.rt_slack_s;
    let t1 = Instant::now();
    // Screening is independent per candidate, which is what makes this worth doing in Rust: the
    // equivalent Python loop was single-threaded and ~40% of the per-file wall clock.
    let mut surv: Vec<(u32, &str)> = (0..lib.nrows)
        .into_par_iter()
        .filter_map(|i| {
            // Decoy peptidoforms carry a "DECOY_" prefix that is not part of the sequence. Failing
            // to strip it makes every decoy untokenisable, which would silently screen targets
            // only and reintroduce the exchangeability bug this stage exists to avoid.
            let pf = pform[i].strip_prefix("DECOY_").unwrap_or(&pform[i]);
            let idx = alpha.tokenise(pf)?;
            let tris = alpha.anchored_tris(&idx);
            if tris.is_empty() {
                return None;
            }
            let c = cid[i] as usize;
            let (lo, hi) = (*lo_by.get(c)?, *hi_by.get(c)?);
            if !lo.is_finite() || !hi.is_finite() {
                return None;
            }
            let b0 = ((lo - slack) / bin).floor() as i64;
            let b1 = ((hi + slack) / bin).floor() as i64;
            let m = pmz[i];
            for w in 0..win.nrows {
                if !(w_lo[w] <= m && m < w_hi[w]) {
                    continue;
                }
                for b in b0..=b1 {
                    if let Some(o) = obs.get(&(w_id[w], b)) {
                        if tris.iter().any(|t| o.contains(t)) {
                            return Some((cid[i], label[i].as_str()));
                        }
                    }
                }
            }
            None
        })
        .collect();
    // Deterministic output: rayon collects in index order, but sorting makes the artifact
    // independent of thread count for byte-comparable reruns.
    surv.sort_unstable();

    let n_t = surv.iter().filter(|(_, l)| *l == "target").count() as u64;
    let n_d = surv.len() as u64 - n_t;
    let ratio = if n_d > 0 {
        n_t as f64 / n_d as f64
    } else {
        f64::NAN
    };
    info!(
        screened = lib.nrows,
        survivors = surv.len(),
        targets = n_t,
        decoys = n_d,
        target_decoy_ratio = ratio,
        elapsed_ms = t1.elapsed().as_millis() as u64,
        "prescan: screened candidates"
    );
    // A one-sided survival means the screen has become label-dependent and the modification's
    // null is gone. Fail loudly rather than emit a library whose modified q-values cannot be
    // estimated.
    if surv.len() > 1000 && (n_t == 0 || n_d == 0) {
        anyhow::bail!(
            "prescan survivors are single-label (targets {n_t}, decoys {n_d}); \
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

    let elapsed = t0.elapsed().as_millis();
    let mut stats: std::collections::BTreeMap<String, serde_json::Value> = Default::default();
    stats.insert("screened".into(), json!(lib.nrows));
    stats.insert("survivors".into(), json!(surv.len()));
    stats.insert("targets".into(), json!(n_t));
    stats.insert("decoys".into(), json!(n_d));
    stats.insert("target_decoy_ratio".into(), json!(ratio));
    stats.insert("index_cells".into(), json!(obs.len()));
    ArtifactReport {
        logical_name: artifact::PRESCAN_SURVIVORS.0.to_string(),
        schema_name: artifact::PRESCAN_SURVIVORS.0.to_string(),
        schema_version: artifact::PRESCAN_SURVIVORS.1,
        stage: "prescan".to_string(),
        rows,
        content_hash: mumdia_io::hash::blake3_file(p.out)?,
        params: json!({
            "ms2": p.ms2,
            "library_precursors": p.library_precursors,
            "run_windows": p.run_windows,
            "tol_da": p.cfg.tol_da,
            "rt_slack_s": p.cfg.rt_slack_s,
            "rt_bin_s": p.cfg.rt_bin_s,
            "top_peaks": p.cfg.top_peaks,
            "mods": p.cfg.mods,
            "anchor_mods": p.cfg.anchor_mods,
            "config_hash": p.config_hash,
        }),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;
    info!(out = p.out, rows, elapsed_ms = elapsed, "prescan: done");
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PrescanConfig {
        PrescanConfig {
            mods: vec!["C:Carbamidomethyl".into(), "M:Oxidation".into()],
            anchor_mods: vec!["C:Farnesyl".into()],
            ..Default::default()
        }
    }

    #[test]
    fn tokenise_merges_i_and_l() {
        let a = Alphabet::build(&cfg()).unwrap();
        // I and L are isobaric, so a mass-delta tag cannot separate them; they must share an index
        // or the screen would look for tags no spectrum can produce.
        assert_eq!(
            a.tokenise("PEPTIDE").unwrap(),
            a.tokenise("PEPTLDE").unwrap()
        );
    }

    #[test]
    fn tokenise_rejects_unknown_modification() {
        let a = Alphabet::build(&cfg()).unwrap();
        assert!(a.tokenise("PEC[Phospho]K").is_none());
        assert!(a.tokenise("PEC[Carbamidomethyl]K").is_some());
    }

    #[test]
    fn anchored_tris_only_cover_the_anchor_and_are_reversible() {
        let a = Alphabet::build(&cfg()).unwrap();
        // Anchor is Farnesyl on C; the CAM'd cysteine must NOT anchor.
        let anchored = a.tokenise("AAC[Farnesyl]AA").unwrap();
        let plain = a.tokenise("AAC[Carbamidomethyl]AA").unwrap();
        let t_anch = a.anchored_tris(&anchored);
        assert!(!t_anch.is_empty());
        assert!(a.anchored_tris(&plain).is_empty());
        // Both orientations present, which is why the screen is blind to reverse decoys.
        for &(x, y, z) in &t_anch {
            assert!(t_anch.contains(&(z, y, x)));
        }
    }

    #[test]
    fn modified_residue_mass_is_backbone_plus_delta() {
        let a = Alphabet::build(&cfg()).unwrap();
        let i = a.modded[&(b'C', "Farnesyl".to_string())];
        let expect = residue_mass(b'C').unwrap() + unimod_mass("Farnesyl").unwrap();
        assert!((a.masses[i] - expect).abs() < 1e-9);
    }

    #[test]
    fn spectrum_trimers_finds_a_planted_ladder() {
        let a = Alphabet::build(&cfg()).unwrap();
        // Build a b-ion ladder for A-A-A: successive deltas of residue mass A.
        let am = residue_mass(b'A').unwrap();
        let mz: Vec<f64> = (0..4).map(|k| 200.0 + k as f64 * am).collect();
        let tri = spectrum_trimers(&mz, &a, 0.005);
        let ia = a.plain[&b'A'] as u32;
        assert!(tri.contains(&(ia, ia, ia)));
    }

    #[test]
    fn bad_mod_spec_is_an_error() {
        assert!(parse_mod_spec("Farnesyl").is_err());
        assert!(parse_mod_spec("CC:Farnesyl").is_err());
        assert!(parse_mod_spec("C:Farnesyl").is_ok());
    }
}
