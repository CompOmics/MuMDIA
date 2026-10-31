//! `localise`: modification site probabilities for the identified peptidoforms
//! (docs/36_ptm_localisation.md). Experimental, opt-in and reporting only.
//!
//! Positional isomers are separate candidates of the search and can all be accepted on the
//! same chromatographic peak. This stage, run on one run's scored table after the final
//! rescore:
//!
//! 1. groups the rows that carry a localised modification into isomer groups (source, base
//!    peptide, modification composition, label) and computes `isomer_group_q`, a picked
//!    target-decoy q over those groups, the unit to count modified peptides in;
//! 2. splits each group's accepted rows (`run_psm_q <= localise.q_max`) into peaks by apex
//!    retention time, per charge; the isomers on one peak share one localisation;
//! 3. enumerates every placement of the modification's count on its candidate residues,
//!    and on the decoy residue, whether or not the library carries that placement;
//! 4. reads every fragment m/z of those placements from the MS2 scans at the apex
//!    (`xic`), and scores the placements by the presence of their site-determining ions
//!    (`model`);
//! 5. writes the posterior, the site probabilities and the model and decoy-residue false
//!    localisation rates to `localisation.parquet`.
//!
//! Nothing upstream reads the table: identifications, q-values and quantities are the same
//! with the stage on or off. Every map that orders a fit or a sum is ordered, and peaks are
//! computed in parallel into a vector in peak order, so the output is deterministic.

pub mod hypotheses;
mod ions;
mod model;
mod xic;

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mumdia_core::config::{LocaliseConfig, ResidueMod};
use mumdia_core::mass::parse_peptidoform;
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table_hashed, Col, TableFile};
use rayon::prelude::*;
use serde_json::json;
use tracing::{info, warn};

use hypotheses::Split;
use ions::N_CLASSES;
use model::Counts;

/// Inputs of [`run`].
pub struct LocaliseParams<'a> {
    /// One run's scored table: a single run's `psms_scored.parquet`, or one run's split of an
    /// experiment (`<run>/scored.parquet`).
    pub scored: &'a str,
    /// That run's `spectra_ms2.parquet`.
    pub spectra_ms2: &'a str,
    /// The run's search-seed `masscal.json` (fragment mass offset and tolerance), if any.
    pub mass_cal: Option<&'a str>,
    /// The `psms_extracted.parquet` the scored rows came from, for each row's apex 1/K0 on
    /// spectra with ion mobility. Without it the apex 1/K0 is read from the spectra.
    pub psms_extracted: Option<&'a str>,
    /// Keep only the rows of this `source`. Required when the table holds several.
    pub source: Option<u32>,
    pub out: &'a str,
    pub cfg: &'a LocaliseConfig,
    /// `peptidoforms.variable_mods`, which give the candidate residues of a modification.
    pub variable_mods: &'a [ResidueMod],
    /// The fragment tolerance when no mass calibration is given (`extract.frag_tol_ppm`).
    pub fallback_tol_ppm: f64,
    pub config_hash: &'a str,
}

/// What [`run`] found, for the orchestrators' log line.
pub struct LocaliseSummary {
    pub written: Written,
    pub rows: u64,
    pub peaks: u64,
    pub ambiguous_target_peaks: u64,
    pub localised_target_peaks: u64,
    pub isomer_groups_at_1pct: u64,
}

/// Peak-level status strings, written to the `status` column.
mod status {
    pub const LOCALISED: &str = "localised";
    pub const AMBIGUOUS: &str = "ambiguous";
    pub const DECOY_RESIDUE: &str = "decoy_residue";
    pub const UNAMBIGUOUS: &str = "unambiguous";
    pub const NOT_ATTEMPTED_Q: &str = "not_attempted_q";
    pub const TOO_MANY: &str = "too_many_hypotheses";
    pub const NO_SPECTRA: &str = "no_spectra";
}

/// One scored row that carries a localised modification.
struct Row {
    /// Row index in the scored table.
    idx: usize,
    candidate_id: u32,
    source: u32,
    pform: String,
    charge: i32,
    decoy: bool,
    protein: String,
    base_peptide_id: u32,
    apex_rt: f64,
    score: f64,
    run_psm_q: f64,
    peak_rank: i32,
    /// Index into `cfg.mods`.
    modi: usize,
    split: Split,
    comp: Vec<i64>,
}

/// One peak: accepted rows of an isomer group and charge that share an apex.
struct Peak {
    rows: Vec<usize>,
    rep: usize,
}

/// What a peak's evidence and model gave.
#[derive(Clone, Debug, Default)]
struct PeakResult {
    status: &'static str,
    hyps: Vec<Vec<usize>>,
    /// Whether each hypothesis uses the decoy residue.
    on_decoy: Vec<bool>,
    post: Vec<f64>,
    site_prob: Vec<f64>,
    /// Candidate positions (real and decoy), ascending.
    positions: Vec<usize>,
    best: Option<usize>,
    min_site_prob: Option<f64>,
    delta_loglik: Option<f64>,
    n_sdi: u32,
    n_sdi_observed: u32,
    n_real: usize,
    n_decoy_positions: usize,
    flr_model: Option<f64>,
    flr_decoy: Option<f64>,
}

/// A peak's raw evidence: per-ion presence and the class counts the rates come from.
struct Evidence {
    common: Counts,
    impossible: Counts,
    /// For each site-determining ion: class, presence, membership.
    sdi: Vec<(usize, bool, Vec<bool>)>,
}

/// Run the stage. See the module documentation.
pub fn run(p: LocaliseParams) -> Result<LocaliseSummary> {
    let t0 = Instant::now();
    let cfg = p.cfg;
    let t = TableFile::open(p.scored).with_context(|| format!("localise: opening {}", p.scored))?;
    let n = t.nrows;
    let source: Vec<u32> = if t.has_column("source") {
        t.u32("source")?
    } else {
        vec![0; n]
    };
    let wanted_source = match p.source {
        Some(s) => s,
        None => {
            let distinct: BTreeSet<u32> = source.iter().copied().collect();
            if distinct.len() > 1 {
                bail!(
                    "localise: {} holds {} sources; give --source (localisation reads one \
                     run's spectra)",
                    p.scored,
                    distinct.len()
                );
            }
            distinct.into_iter().next().unwrap_or(0)
        }
    };
    let pform = t.str("peptidoform")?;
    // Parse once; keep only rows of the source carrying a localised modification.
    let mut keep: Vec<(usize, usize, Split)> = Vec::new();
    for i in 0..n {
        if source[i] != wanted_source {
            continue;
        }
        for (mi, name) in cfg.mods.iter().enumerate() {
            if let Some(s) = hypotheses::split(&pform[i], name) {
                keep.push((i, mi, s));
                break;
            }
        }
    }
    let idx: Vec<usize> = keep.iter().map(|k| k.0).collect();
    let candidate_id = t.u32_rows("candidate_id", &idx)?;
    let charge = t.i32_rows("charge", &idx)?;
    let label = t.str_rows("label", &idx)?;
    let protein = t.str_rows("protein", &idx)?;
    let bpid = t.u32_rows("base_peptide_id", &idx)?;
    let apex_rt = t.f64_rows("apex_rt", &idx)?;
    let score = t.f64_rows("score", &idx)?;
    let q_col = if t.has_column("run_psm_q") {
        "run_psm_q"
    } else {
        "q_value"
    };
    let run_q = t.f64_rows(q_col, &idx)?;
    let peak_rank = if t.has_column("selected_peak_rank") {
        t.i32_rows("selected_peak_rank", &idx)?
    } else {
        vec![0; idx.len()]
    };
    let rows: Vec<Row> = keep
        .into_iter()
        .enumerate()
        .map(|(r, (i, modi, split))| Row {
            idx: i,
            candidate_id: candidate_id[r],
            source: wanted_source,
            comp: hypotheses::composition_key(&pform[i]).unwrap_or_default(),
            pform: pform[i].clone(),
            charge: charge[r],
            decoy: label[r] != "target",
            protein: protein[r].clone(),
            base_peptide_id: bpid[r],
            apex_rt: apex_rt[r],
            score: score[r],
            run_psm_q: run_q[r],
            peak_rank: peak_rank[r],
            modi,
            split,
        })
        .collect();
    drop(pform);

    // Candidate residues per modification: the configured variable modifications, else the
    // residues the scored table carries it on.
    let candidates: Vec<Vec<u8>> = cfg
        .mods
        .iter()
        .enumerate()
        .map(|(mi, name)| {
            let mut set: BTreeSet<u8> = p
                .variable_mods
                .iter()
                .filter(|v| &v.name == name)
                .filter_map(|v| u8::try_from(v.residue).ok())
                .map(|c| c.to_ascii_uppercase())
                .collect();
            if set.is_empty() {
                for r in rows.iter().filter(|r| r.modi == mi) {
                    for &s in &r.split.sites {
                        set.insert(r.split.residues[s]);
                    }
                }
            }
            set.into_iter().collect()
        })
        .collect();
    let decoy_residue: Vec<Option<u8>> = candidates
        .iter()
        .zip(&cfg.mods)
        .map(|(c, name)| {
            let d = cfg.decoy_residue.as_bytes().first().copied()?;
            if c.contains(&d) {
                warn!(
                    modification = %name,
                    residue = %(d as char),
                    "localise: the decoy residue carries the modification in this table; no \
                     decoy placements for it"
                );
                return None;
            }
            Some(d)
        })
        .collect();

    // --- isomer groups and their picked target-decoy q ---
    let mut group_ids: BTreeMap<(u32, u32, Vec<i64>, bool), u32> = BTreeMap::new();
    let mut q_keys: BTreeMap<(u32, u32, Vec<i64>), u32> = BTreeMap::new();
    for r in &rows {
        let gk = (r.source, r.base_peptide_id, r.comp.clone(), r.decoy);
        let next = group_ids.len() as u32;
        group_ids.entry(gk).or_insert(next);
        let qk = (r.source, r.base_peptide_id, r.comp.clone());
        let next = q_keys.len() as u32;
        q_keys.entry(qk).or_insert(next);
    }
    let group_of: Vec<u32> = rows
        .iter()
        .map(|r| group_ids[&(r.source, r.base_peptide_id, r.comp.clone(), r.decoy)])
        .collect();
    let qkey_of: Vec<u32> = rows
        .iter()
        .map(|r| q_keys[&(r.source, r.base_peptide_id, r.comp.clone())])
        .collect();
    let is_decoy: Vec<bool> = rows.iter().map(|r| r.decoy).collect();
    let none = vec![false; rows.len()];
    let isomer_group_q = crate::stages::rescore::grouped_q(
        &qkey_of,
        &rows.iter().map(|r| r.score).collect::<Vec<_>>(),
        &is_decoy,
        &none,
        &none,
        crate::stages::rescore::QMode::Decoy,
        1.0,
    );

    // --- peaks: accepted rows of a group and charge, split by apex ---
    let mut by_gc: BTreeMap<(u32, i32), Vec<usize>> = BTreeMap::new();
    for (r, row) in rows.iter().enumerate() {
        if row.run_psm_q <= cfg.q_max {
            by_gc.entry((group_of[r], row.charge)).or_default().push(r);
        }
    }
    let mut peaks: Vec<Peak> = Vec::new();
    for (_, mut members) in by_gc {
        members.sort_by(|&a, &b| {
            rows[a]
                .apex_rt
                .total_cmp(&rows[b].apex_rt)
                .then(rows[a].idx.cmp(&rows[b].idx))
        });
        let mut cur: Vec<usize> = Vec::new();
        for r in members {
            if let Some(&first) = cur.first() {
                if rows[r].apex_rt - rows[first].apex_rt > cfg.peak_rt_tol_s {
                    peaks.push(peak_of(&rows, std::mem::take(&mut cur)));
                }
            }
            cur.push(r);
        }
        if !cur.is_empty() {
            peaks.push(peak_of(&rows, cur));
        }
    }
    let mut peak_of_row: Vec<Option<u32>> = vec![None; rows.len()];
    for (pi, pk) in peaks.iter().enumerate() {
        for &r in &pk.rows {
            peak_of_row[r] = Some(pi as u32);
        }
    }

    // --- hypotheses per peak ---
    let mut results: Vec<PeakResult> = peaks
        .iter()
        .map(|pk| {
            let r = &rows[pk.rep];
            let dec = decoy_residue[r.modi];
            let positions = hypotheses::candidate_positions(&r.split, &candidates[r.modi], dec);
            let k = r.split.sites.len();
            let n_decoy_positions = positions
                .iter()
                .filter(|&&i| Some(r.split.residues[i]) == dec)
                .count();
            match hypotheses::enumerate(&positions, k, cfg.max_hypotheses) {
                None => PeakResult {
                    status: status::TOO_MANY,
                    positions,
                    n_decoy_positions,
                    ..Default::default()
                },
                Some(hyps) => {
                    let on_decoy: Vec<bool> = hyps
                        .iter()
                        .map(|h| h.iter().any(|&i| Some(r.split.residues[i]) == dec))
                        .collect();
                    let n_real = on_decoy.iter().filter(|&&d| !d).count();
                    PeakResult {
                        status: "",
                        hyps,
                        on_decoy,
                        positions,
                        n_real,
                        n_decoy_positions,
                        ..Default::default()
                    }
                }
            }
        })
        .collect();
    // Peaks whose placement is fixed by the sequence need no evidence.
    for (pi, res) in results.iter_mut().enumerate() {
        if res.status.is_empty() && res.n_real <= 1 {
            let rep = &rows[peaks[pi].rep];
            res.status = status::UNAMBIGUOUS;
            let real = res.on_decoy.iter().position(|&d| !d);
            let mut sp = vec![0.0; rep.split.residues.len()];
            let mut post = vec![0.0; res.hyps.len()];
            if let Some(h) = real {
                post[h] = 1.0;
                for &i in &res.hyps[h] {
                    sp[i] = 1.0;
                }
                res.min_site_prob = Some(1.0);
            }
            res.best = real;
            res.post = post;
            res.site_prob = sp;
        }
    }

    // --- evidence from the spectra ---
    let need: Vec<usize> = (0..peaks.len())
        .filter(|&pi| results[pi].status.is_empty())
        .collect();
    let mut global_d = [0.5; N_CLASSES];
    let mut global_f = [0.5; N_CLASSES];
    let mut tol_used = p.fallback_tol_ppm;
    if !need.is_empty() {
        let scans = crate::spectra::load_ms2(p.spectra_ms2)
            .with_context(|| format!("localise: loading {}", p.spectra_ms2))?;
        let has_im = scans.iter().any(|s| s.im.is_some());
        let index = xic::ScanIndex::new(&scans);
        let (mass, tol) = crate::masscal::read_mass_cal(p.mass_cal, p.fallback_tol_ppm)?;
        tol_used = tol;
        let apex_im: BTreeMap<(u32, i32), f64> = match (has_im, p.psms_extracted) {
            (true, Some(path)) if std::path::Path::new(path).exists() => {
                let e = TableFile::open(path)?;
                let cid = e.u32("candidate_id")?;
                let rank = if e.has_column("peak_rank") {
                    e.i32("peak_rank")?
                } else {
                    vec![0; e.nrows]
                };
                let im = if e.has_column("apex_im") {
                    e.opt_f64("apex_im")?
                } else {
                    vec![None; e.nrows]
                };
                (0..e.nrows)
                    .filter_map(|i| {
                        im[i]
                            .filter(|v| v.is_finite())
                            .map(|v| ((cid[i], rank[i]), v))
                    })
                    .collect()
            }
            _ => BTreeMap::new(),
        };
        let evidence: Vec<Option<Evidence>> = need
            .par_iter()
            .map(|&pi| {
                let r = &rows[peaks[pi].rep];
                let res = &results[pi];
                let prec_mz = parse_peptidoform(&r.pform)
                    .map(|pp| pp.precursor_mz(r.charge))
                    .unwrap_or(f64::NAN);
                let zmax = if r.charge >= 3 { 2 } else { 1 };
                let set = ions::build(&r.split, &res.hyps, zmax);
                let known_im = if has_im {
                    apex_im.get(&(r.candidate_id, r.peak_rank)).copied()
                } else {
                    None
                };
                let sel = index.apex_scans(&scans, prec_mz, r.apex_rt, known_im, cfg.apex_scans);
                if sel.is_empty() {
                    return None;
                }
                let im = if has_im {
                    known_im.or_else(|| {
                        let common: Vec<f64> = (0..set.ions.len())
                            .filter(|&j| set.is_common(j))
                            .map(|j| set.ions[j].mz)
                            .collect();
                        xic::apex_im_of(&scans, &sel, &common, &mass, tol)
                    })
                } else {
                    None
                };
                let band = im.map(|c| (c, cfg.im_half_width));
                let mut ev = Evidence {
                    common: Counts::default(),
                    impossible: Counts::default(),
                    sdi: Vec::new(),
                };
                for (j, ion) in set.ions.iter().enumerate() {
                    let here = xic::present(&scans, &sel, ion.mz, &mass, tol, band);
                    if set.is_common(j) {
                        ev.common.add(ion.class, here);
                    } else {
                        ev.sdi.push((ion.class, here, ion.member.clone()));
                    }
                }
                for &(mz, class) in &set.impossible {
                    ev.impossible
                        .add(class, xic::present(&scans, &sel, mz, &mass, tol, band));
                }
                Some(ev)
            })
            .collect();
        // Run-wide rates from the target peaks, summed in peak order.
        let (mut cd, mut cf) = (Counts::default(), Counts::default());
        for (k, &pi) in need.iter().enumerate() {
            if let (Some(ev), false) = (&evidence[k], rows[peaks[pi].rep].decoy) {
                cd.merge(&ev.common);
                cf.merge(&ev.impossible);
            }
        }
        global_d = cd.laplace();
        global_f = cf.laplace();
        let scored: Vec<(usize, PeakResult)> = need
            .par_iter()
            .zip(evidence.par_iter())
            .map(|(&pi, ev)| {
                let mut res = results[pi].clone();
                let Some(ev) = ev else {
                    res.status = status::NO_SPECTRA;
                    return (pi, res);
                };
                let d = ev.common.shrunk(&global_d, cfg.prior_strength);
                let f = ev.impossible.shrunk(&global_f, cfg.prior_strength);
                let obs: Vec<model::Observed> = ev
                    .sdi
                    .iter()
                    .map(|(c, here, m)| model::Observed {
                        class: *c,
                        present: *here,
                        member: m,
                    })
                    .collect();
                let ll = model::loglik(res.hyps.len(), &obs, &d, &f);
                let post = model::posterior(&ll, cfg.temperature);
                let rep = &rows[peaks[pi].rep];
                let sp = model::site_probabilities(rep.split.residues.len(), &res.hyps, &post);
                let b = model::best(&res.hyps, &post);
                let min_sp = res.hyps[b]
                    .iter()
                    .map(|&i| sp[i])
                    .fold(f64::INFINITY, f64::min);
                let runner_up = (0..ll.len())
                    .filter(|&h| h != b)
                    .map(|h| ll[h])
                    .fold(f64::NEG_INFINITY, f64::max);
                res.delta_loglik = Some(if runner_up.is_finite() {
                    ll[b] - runner_up
                } else {
                    0.0
                });
                res.n_sdi = ev.sdi.len() as u32;
                res.n_sdi_observed = ev.sdi.iter().filter(|x| x.1).count() as u32;
                res.status = if res.on_decoy[b] {
                    status::DECOY_RESIDUE
                } else if min_sp >= cfg.threshold {
                    status::LOCALISED
                } else {
                    status::AMBIGUOUS
                };
                res.best = Some(b);
                res.min_site_prob = Some(min_sp);
                res.post = post;
                res.site_prob = sp;
                (pi, res)
            })
            .collect();
        for (pi, res) in scored {
            results[pi] = res;
        }
    }

    // --- false localisation rates over the target peaks with a real choice ---
    let flr_rows: Vec<usize> = (0..peaks.len())
        .filter(|&pi| {
            let res = &results[pi];
            !rows[peaks[pi].rep].decoy
                && res.n_real >= 2
                && res.min_site_prob.is_some()
                && matches!(
                    res.status,
                    status::LOCALISED | status::AMBIGUOUS | status::DECOY_RESIDUE
                )
        })
        .collect();
    let (real_pos, decoy_pos) = flr_rows.iter().fold((0usize, 0usize), |acc, &pi| {
        let res = &results[pi];
        (
            acc.0 + res.positions.len() - res.n_decoy_positions,
            acc.1 + res.n_decoy_positions,
        )
    });
    let ratio = if decoy_pos > 0 {
        real_pos as f64 / decoy_pos as f64
    } else {
        f64::NAN
    };
    let flr = flr_curve(
        &flr_rows
            .iter()
            .map(|&pi| {
                let res = &results[pi];
                (
                    res.min_site_prob.unwrap_or(0.0),
                    res.status == status::DECOY_RESIDUE,
                )
            })
            .collect::<Vec<_>>(),
        ratio,
    );
    for (k, &pi) in flr_rows.iter().enumerate() {
        results[pi].flr_model = Some(flr[k].0);
        results[pi].flr_decoy = flr[k].1;
    }

    // --- the table ---
    let m = rows.len();
    let mut c_status = Vec::with_capacity(m);
    let mut c_peak = Vec::with_capacity(m);
    let mut c_rep = Vec::with_capacity(m);
    let mut c_best = Vec::with_capacity(m);
    let mut c_sites = Vec::with_capacity(m);
    let mut c_min = Vec::with_capacity(m);
    let mut c_own_prob = Vec::with_capacity(m);
    let mut c_own_site = Vec::with_capacity(m);
    let mut c_loc = Vec::with_capacity(m);
    let mut c_ondecoy = Vec::with_capacity(m);
    let mut c_nhyp = Vec::with_capacity(m);
    let mut c_nsdi = Vec::with_capacity(m);
    let mut c_nobs = Vec::with_capacity(m);
    let mut c_dll = Vec::with_capacity(m);
    let mut c_flrm = Vec::with_capacity(m);
    let mut c_flrd = Vec::with_capacity(m);
    for (r, row) in rows.iter().enumerate() {
        let Some(pi) = peak_of_row[r] else {
            c_status.push(status::NOT_ATTEMPTED_Q.to_string());
            c_peak.push(None);
            c_rep.push(false);
            c_best.push(None);
            c_sites.push(None);
            c_min.push(None);
            c_own_prob.push(None);
            c_own_site.push(None);
            c_loc.push(false);
            c_ondecoy.push(false);
            c_nhyp.push(0);
            c_nsdi.push(0);
            c_nobs.push(0);
            c_dll.push(None);
            c_flrm.push(None);
            c_flrd.push(None);
            continue;
        };
        let pk = &peaks[pi as usize];
        let res = &results[pi as usize];
        let rep = &rows[pk.rep];
        let name = &cfg.mods[rep.modi];
        c_status.push(res.status.to_string());
        c_peak.push(Some(pi as i32));
        c_rep.push(pk.rep == r);
        c_nhyp.push(res.hyps.len() as u32);
        c_nsdi.push(res.n_sdi);
        c_nobs.push(res.n_sdi_observed);
        c_dll.push(res.delta_loglik);
        c_flrm.push(res.flr_model);
        c_flrd.push(res.flr_decoy);
        c_min.push(res.min_site_prob);
        match res.best {
            Some(b) if !res.post.is_empty() => {
                c_best.push(Some(hypotheses::render(
                    &rep.pform,
                    &rep.split,
                    name,
                    &res.hyps[b],
                )));
                c_sites.push(Some(site_string(
                    &rep.split,
                    &res.positions,
                    &res.site_prob,
                )));
                let own = res.hyps.iter().position(|h| *h == row.split.sites);
                c_own_prob.push(own.map(|h| res.post[h]));
                let own_site = if row.split.sites.iter().all(|i| res.positions.contains(i)) {
                    Some(
                        row.split
                            .sites
                            .iter()
                            .map(|&i| res.site_prob.get(i).copied().unwrap_or(0.0))
                            .fold(f64::INFINITY, f64::min),
                    )
                } else {
                    None
                };
                c_own_site.push(own_site);
                c_loc.push(matches!(
                    res.status,
                    status::LOCALISED | status::UNAMBIGUOUS
                ));
                c_ondecoy.push(res.on_decoy[b]);
            }
            _ => {
                c_best.push(None);
                c_sites.push(None);
                c_own_prob.push(None);
                c_own_site.push(None);
                c_loc.push(false);
                c_ondecoy.push(false);
            }
        }
    }
    let n_groups_1pct = rows
        .iter()
        .zip(&isomer_group_q)
        .filter(|(r, &q)| !r.decoy && q <= 0.01)
        .count() as u64;
    let written = write_table_hashed(
        p.out,
        vec![
            Col::U32(
                "candidate_id".into(),
                rows.iter().map(|r| r.candidate_id).collect(),
            ),
            Col::U32("source".into(), rows.iter().map(|r| r.source).collect()),
            Col::Str(
                "peptidoform".into(),
                rows.iter().map(|r| r.pform.clone()).collect(),
            ),
            Col::I32("charge".into(), rows.iter().map(|r| r.charge).collect()),
            Col::Str(
                "label".into(),
                rows.iter()
                    .map(|r| if r.decoy { "decoy" } else { "target" }.to_string())
                    .collect(),
            ),
            Col::Str(
                "protein".into(),
                rows.iter().map(|r| r.protein.clone()).collect(),
            ),
            Col::F64("apex_rt".into(), rows.iter().map(|r| r.apex_rt).collect()),
            Col::F64("score".into(), rows.iter().map(|r| r.score).collect()),
            Col::F64(
                "run_psm_q".into(),
                rows.iter().map(|r| r.run_psm_q).collect(),
            ),
            Col::Str(
                "localised_mod".into(),
                rows.iter().map(|r| cfg.mods[r.modi].clone()).collect(),
            ),
            Col::U32("isomer_group_id".into(), group_of),
            Col::F64("isomer_group_q".into(), isomer_group_q),
            Col::OptI32("isomer_peak_id".into(), c_peak),
            Col::Bool("is_peak_representative".into(), c_rep),
            Col::U32("n_hypotheses".into(), c_nhyp),
            Col::OptStr("best_site_peptidoform".into(), c_best),
            Col::OptStr("site_probabilities".into(), c_sites),
            Col::OptF64("min_site_prob".into(), c_min),
            Col::OptF64("peptidoform_probability".into(), c_own_prob),
            Col::OptF64("own_site_prob".into(), c_own_site),
            Col::Bool("localised".into(), c_loc),
            Col::Bool("on_decoy_residue".into(), c_ondecoy),
            Col::U32("n_sdi".into(), c_nsdi),
            Col::U32("n_sdi_observed".into(), c_nobs),
            Col::OptF64("delta_loglik".into(), c_dll),
            Col::OptF64("site_flr_model".into(), c_flrm),
            Col::OptF64("site_flr_decoy".into(), c_flrd),
            Col::Str("status".into(), c_status),
        ],
    )?;

    // --- report ---
    let target_peaks = || {
        peaks
            .iter()
            .enumerate()
            .filter(|(_, pk)| !rows[pk.rep].decoy)
    };
    let count_status = |s: &str| {
        target_peaks()
            .filter(|(pi, _)| results[*pi].status == s)
            .count() as u64
    };
    let ambiguous = flr_rows.len() as u64;
    let localised_amb = flr_rows
        .iter()
        .filter(|&&pi| results[pi].status == status::LOCALISED)
        .count() as u64;
    // The rates at the threshold: those of the lowest probability still at or above it,
    // which are the largest, the curves being monotone.
    let at_threshold = flr_rows
        .iter()
        .zip(&flr)
        .filter(|(&pi, _)| results[pi].min_site_prob.unwrap_or(0.0) >= cfg.threshold)
        .map(|(_, f)| *f)
        .reduce(|a, b| {
            (
                a.0.max(b.0),
                match (a.1, b.1) {
                    (Some(x), Some(y)) => Some(x.max(y)),
                    (x, y) => x.or(y),
                },
            )
        });
    let mut stats = BTreeMap::new();
    stats.insert("rows".into(), json!(m));
    stats.insert("peaks".into(), json!(peaks.len()));
    stats.insert("target_peaks".into(), json!(target_peaks().count()));
    stats.insert(
        "target_peaks_localised".into(),
        json!(count_status(status::LOCALISED)),
    );
    stats.insert(
        "target_peaks_ambiguous".into(),
        json!(count_status(status::AMBIGUOUS)),
    );
    stats.insert(
        "target_peaks_decoy_residue".into(),
        json!(count_status(status::DECOY_RESIDUE)),
    );
    stats.insert(
        "target_peaks_unambiguous".into(),
        json!(count_status(status::UNAMBIGUOUS)),
    );
    stats.insert(
        "target_peaks_too_many_hypotheses".into(),
        json!(count_status(status::TOO_MANY)),
    );
    stats.insert(
        "target_peaks_no_spectra".into(),
        json!(count_status(status::NO_SPECTRA)),
    );
    stats.insert("target_peaks_with_a_choice".into(), json!(ambiguous));
    stats.insert(
        "localisation_rate".into(),
        json!(if ambiguous > 0 {
            localised_amb as f64 / ambiguous as f64
        } else {
            f64::NAN
        }),
    );
    stats.insert(
        "site_flr_model_at_threshold".into(),
        json!(at_threshold.map(|f| f.0)),
    );
    stats.insert(
        "site_flr_decoy_at_threshold".into(),
        json!(at_threshold.and_then(|f| f.1)),
    );
    stats.insert("candidate_to_decoy_residue_ratio".into(), json!(ratio));
    stats.insert("detection_rate_by_class".into(), json!(global_d));
    stats.insert("false_presence_rate_by_class".into(), json!(global_f));
    stats.insert("isomer_groups_at_1pct".into(), json!(n_groups_1pct));
    stats.insert("fragment_tol_ppm".into(), json!(tol_used));
    let elapsed = t0.elapsed().as_millis();
    ArtifactReport {
        logical_name: artifact::LOCALISATION.0.to_string(),
        schema_name: artifact::LOCALISATION.0.to_string(),
        schema_version: artifact::LOCALISATION.1,
        stage: "localise".to_string(),
        rows: written.rows,
        content_hash: written.content_hash.clone(),
        params: json!({
            "config_hash": p.config_hash,
            "scored": p.scored,
            "spectra_ms2": p.spectra_ms2,
            "mass_cal": p.mass_cal,
            "psms_extracted": p.psms_extracted,
            "source": wanted_source,
            "mods": cfg.mods,
            "candidate_residues": candidates
                .iter()
                .map(|c| String::from_utf8_lossy(c).into_owned())
                .collect::<Vec<_>>(),
            "decoy_residue": cfg.decoy_residue,
            "threshold": cfg.threshold,
            "q_max": cfg.q_max,
            "max_hypotheses": cfg.max_hypotheses,
            "peak_rt_tol_s": cfg.peak_rt_tol_s,
            "apex_scans": cfg.apex_scans,
            "im_half_width": cfg.im_half_width,
            "prior_strength": cfg.prior_strength,
            "temperature": cfg.temperature,
        }),
        stats,
        model_identity: Some("localise-sdi-presence-v1".to_string()),
        elapsed_ms: elapsed,
    }
    .write_for(p.out)?;
    info!(
        out = p.out,
        rows = m,
        peaks = peaks.len(),
        with_a_choice = ambiguous,
        localised = localised_amb,
        isomer_groups_at_1pct = n_groups_1pct,
        elapsed_ms = elapsed as u64,
        "localise: done"
    );
    Ok(LocaliseSummary {
        written,
        rows: m as u64,
        peaks: peaks.len() as u64,
        ambiguous_target_peaks: ambiguous,
        localised_target_peaks: localised_amb,
        isomer_groups_at_1pct: n_groups_1pct,
    })
}

/// A peak of `members` with its representative: the highest score, ties to the earlier row.
fn peak_of(rows: &[Row], members: Vec<usize>) -> Peak {
    let mut rep = members[0];
    for &r in &members[1..] {
        if rows[r].score > rows[rep].score
            || (rows[r].score == rows[rep].score && rows[r].idx < rows[rep].idx)
        {
            rep = r;
        }
    }
    Peak { rows: members, rep }
}

/// `S3:0.9120;T5:0.0710;A6:0.0170`: every candidate position (1-based) with its site
/// probability.
fn site_string(s: &Split, positions: &[usize], sp: &[f64]) -> String {
    positions
        .iter()
        .map(|&i| format!("{}{}:{:.4}", s.residues[i] as char, i + 1, sp[i]))
        .collect::<Vec<_>>()
        .join(";")
}

/// The false localisation rates at each peak's own minimum site probability, for peaks
/// `(min_site_prob, best_on_decoy)`: the model's (the mean `1 - p` of the real localisations
/// at or above it) and the decoy residue's (decoy wins at or above it, times the
/// candidate-to-decoy residue ratio, over the real localisations at or above it). Both are
/// made monotone the way q-values are: each is the minimum over every lower threshold.
pub(crate) fn flr_curve(peaks: &[(f64, bool)], ratio: f64) -> Vec<(f64, Option<f64>)> {
    let mut order: Vec<usize> = (0..peaks.len()).collect();
    order.sort_by(|&a, &b| peaks[b].0.total_cmp(&peaks[a].0).then(a.cmp(&b)));
    let mut raw: Vec<(f64, f64)> = vec![(0.0, f64::NAN); peaks.len()];
    let (mut n_real, mut n_decoy, mut err) = (0usize, 0usize, 0.0f64);
    let mut k = 0;
    while k < order.len() {
        // a block of tied probabilities shares one value
        let p0 = peaks[order[k]].0;
        let mut e = k;
        while e < order.len() && peaks[order[e]].0 == p0 {
            let (p, on_decoy) = peaks[order[e]];
            if on_decoy {
                n_decoy += 1;
            } else {
                n_real += 1;
                err += 1.0 - p;
            }
            e += 1;
        }
        let model = if n_real > 0 { err / n_real as f64 } else { 0.0 };
        let decoy = if ratio.is_finite() {
            n_decoy as f64 * ratio / n_real.max(1) as f64
        } else {
            f64::NAN
        };
        for &i in &order[k..e] {
            raw[i] = (model, decoy);
        }
        k = e;
    }
    // monotone: walk from the lowest probability up, keeping the running minimum
    let (mut mm, mut md) = (f64::INFINITY, f64::INFINITY);
    let mut out = vec![(0.0, None); peaks.len()];
    for &i in order.iter().rev() {
        mm = mm.min(raw[i].0);
        out[i].0 = mm;
        if raw[i].1.is_finite() {
            md = md.min(raw[i].1);
            out[i].1 = Some(md);
        }
    }
    out
}

#[cfg(test)]
mod tests;
