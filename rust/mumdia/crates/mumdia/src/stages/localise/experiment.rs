//! The `localise` stage over the runs of an experiment (docs/36_ptm_localisation.md
//! section 5).
//!
//! Every run is analysed up to its log-likelihoods first, so that one temperature serves all
//! of them (fitted on the pooled runs under `"auto"`), and the per-run tables are then written
//! as [`super::run`] writes them. With `localise.combine_runs` the runs are then combined:
//!
//! 1. every run's confident target apexes are aligned onto a reference run by the
//!    binned-median curves of the second pass (`second_pass::alignment`);
//! 2. the peaks of one precursor (localised modification, the peptidoform without it, its
//!    count, label and charge) are clustered across runs on the aligned scale: in order of
//!    descending representative score, a peak joins the first consensus peak without a peak
//!    of its run whose anchor's apex lies inside its aligned elution bounds or whose bounds
//!    hold its apex, and otherwise starts one; a run where the precursor is absent, or
//!    observed only at another retention time, contributes nothing to that consensus peak;
//! 3. the log-likelihoods of the runs whose peak joined are summed hypothesis by hypothesis
//!    (runs in source order; a run whose hypothesis list differs is left out), and the
//!    posterior is their softmax at the experiment temperature, fitted on the consensus peaks
//!    as the run temperature is on the run peaks, its prior centred on the run temperature
//!    times the square root of the mean number of runs summed (the geometric mean of
//!    independent runs, the run temperature, and of runs that repeat each other's evidence,
//!    that times the number of runs);
//! 4. `localisation_experiment.parquet` holds one row per consensus peak, with the false
//!    localisation rates over the consensus peaks.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{bail, Result};
use mumdia_core::config::LocaliseConfig;
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table_hashed, Col, TableFile};
use serde_json::json;
use tracing::info;

use super::{
    analyse, calibrate, finish, flr_curve, hypotheses, model, site_string, status, write,
    LocaliseParams, LocaliseSummary, RunAnalysis, Temperature,
};

/// Inputs of [`run_experiment`].
pub struct ExperimentLocaliseParams<'a> {
    /// One entry per run, each with its own `source`, spectra and output table. Every entry
    /// carries the same configuration.
    pub runs: Vec<LocaliseParams<'a>>,
    /// Where the experiment-level table goes; `None` writes none.
    pub out: Option<&'a str>,
    pub config_hash: &'a str,
}

/// What [`run_experiment`] found.
pub struct ExperimentLocaliseSummary {
    pub runs: Vec<LocaliseSummary>,
    /// The run temperature, shared by every run.
    pub temperature: f64,
    pub combined: Option<CombineSummary>,
}

/// What the combination wrote.
pub struct CombineSummary {
    pub written: Written,
    pub consensus_peaks: u64,
    /// Target consensus peaks with a real choice of site, and of them localised.
    pub with_a_choice: u64,
    pub localised: u64,
    pub temperature: f64,
}

/// Run the stage on every run of an experiment and combine them. See the module
/// documentation.
pub fn run_experiment(p: ExperimentLocaliseParams) -> Result<ExperimentLocaliseSummary> {
    let t0 = Instant::now();
    let Some(first) = p.runs.first() else {
        bail!("localise: an experiment needs at least one run");
    };
    let cfg = first.cfg;
    let mut analyses: Vec<RunAnalysis> = Vec::with_capacity(p.runs.len());
    for r in &p.runs {
        info!(scored = r.scored, "localise: reading the evidence of a run");
        analyses.push(analyse(r)?);
    }
    let temp = match cfg.fixed_temperature() {
        Some(t) => Temperature::fixed(t),
        None => {
            let refs: Vec<&RunAnalysis> = analyses.iter().collect();
            calibrate::fit_runs(&refs, 1.0)
        }
    };
    let mut runs = Vec::with_capacity(analyses.len());
    for (a, r) in analyses.iter_mut().zip(&p.runs) {
        finish(a, cfg, temp.value);
        let ms = a.analyse_ms;
        runs.push(write(a, r, &temp, ms)?);
    }
    let combined = match p.out {
        Some(out) if cfg.combine_runs && analyses.len() > 1 => Some(combine(
            &analyses,
            &p.runs,
            out,
            cfg,
            &temp,
            p.config_hash,
            t0,
        )?),
        _ => None,
    };
    Ok(ExperimentLocaliseSummary {
        runs,
        temperature: temp.value,
        combined,
    })
}

/// One run's peak, placed on the reference run's retention-time scale.
struct Observation {
    run: usize,
    peak: usize,
    apex: f64,
    lo: f64,
    hi: f64,
    score: f64,
}

/// One consensus peak.
struct Consensus {
    /// The anchor: the highest-scoring observation, `(run, peak)`.
    anchor: (usize, usize),
    /// Every observation, `(run, peak)`, in run order.
    members: Vec<(usize, usize)>,
    /// The runs whose log-likelihoods were summed, in run order.
    summed: Vec<usize>,
    /// Their site-determining ions, and how many were present.
    sdi: (u32, u32),
    /// The anchor's apex on the reference scale.
    rt: f64,
    ll: Vec<f64>,
    /// Index of the member whose hypotheses `ll` is over (the anchor when it was scored).
    hyps_of: Option<(usize, usize)>,
    best: Option<usize>,
    min_site_prob: Option<f64>,
    post: Vec<f64>,
    site_prob: Vec<f64>,
    status: &'static str,
    delta_loglik: Option<f64>,
    flr_model: Option<f64>,
    flr_decoy: Option<f64>,
}

/// The confident target apexes `(source, candidate_id, apex_rt)` of a run's scored table.
fn confident_apexes(p: &LocaliseParams, source: u32) -> Result<Vec<(u32, u32, f64)>> {
    let t = TableFile::open(p.scored)?;
    let cid = t.u32("candidate_id")?;
    let label = t.str("label")?;
    let apex = t.f64("apex_rt")?;
    let q = if t.has_column("run_psm_q") {
        t.f64("run_psm_q")?
    } else {
        t.f64("q_value")?
    };
    let src = if t.has_column("source") {
        t.u32("source")?
    } else {
        vec![source; t.nrows]
    };
    Ok((0..t.nrows)
        .filter(|&i| src[i] == source && label[i] == "target" && q[i] <= 0.01)
        .filter(|&i| apex[i].is_finite())
        .map(|i| (source, cid[i], apex[i]))
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn combine(
    analyses: &[RunAnalysis],
    params: &[LocaliseParams],
    out: &str,
    cfg: &LocaliseConfig,
    temp: &Temperature,
    config_hash: &str,
    t0: Instant,
) -> Result<CombineSummary> {
    // --- every run on the reference run's retention-time scale ---
    let mut obs: Vec<(u32, u32, f64)> = Vec::new();
    for (a, p) in analyses.iter().zip(params) {
        obs.extend(confident_apexes(p, a.source)?);
    }
    let (reference, curves) = crate::stages::second_pass::alignment(&obs);
    let to_ref = |source: u32, x: f64| curves.get(&source).map_or(x, |c| c.map(x));

    // --- the peaks of each precursor, across runs ---
    type Key = (usize, String, usize, bool, i32);
    let mut by_key: BTreeMap<Key, Vec<Observation>> = BTreeMap::new();
    for (ri, a) in analyses.iter().enumerate() {
        for (pi, pk) in a.peaks.iter().enumerate() {
            let rep = &a.rows[pk.rep];
            let base = hypotheses::render(&rep.pform, &rep.split, &cfg.mods[rep.modi], &[]);
            let key = (rep.modi, base, rep.split.sites.len(), rep.decoy, rep.charge);
            let apex = to_ref(a.source, rep.apex_rt);
            let (lo, hi) = match pk.bounds {
                Some((l, h)) => (to_ref(a.source, l), to_ref(a.source, h)),
                None => (apex - cfg.peak_rt_tol_s, apex + cfg.peak_rt_tol_s),
            };
            by_key.entry(key).or_default().push(Observation {
                run: ri,
                peak: pi,
                apex,
                lo: lo.min(apex),
                hi: hi.max(apex),
                score: rep.score,
            });
        }
    }
    let mut cons: Vec<Consensus> = Vec::new();
    for (_, mut o) in by_key {
        o.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.run.cmp(&b.run))
                .then(a.peak.cmp(&b.peak))
        });
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        for (oi, x) in o.iter().enumerate() {
            let join = clusters.iter().position(|c| {
                let an = &o[c[0]];
                !c.iter().any(|&m| o[m].run == x.run)
                    && ((an.lo <= x.apex && x.apex <= an.hi)
                        || (x.lo <= an.apex && an.apex <= x.hi))
            });
            match join {
                Some(ci) => clusters[ci].push(oi),
                None => clusters.push(vec![oi]),
            }
        }
        for c in clusters {
            let an = &o[c[0]];
            let mut members: Vec<(usize, usize)> =
                c.iter().map(|&m| (o[m].run, o[m].peak)).collect();
            members.sort_unstable();
            let anchor = (an.run, an.peak);
            // the hypotheses the sum is over: the anchor's, else the first scored member's
            let scored = |&(r, p): &(usize, usize)| !analyses[r].results[p].ll.is_empty();
            let hyps_of = if scored(&anchor) {
                Some(anchor)
            } else {
                members.iter().copied().find(|m| scored(m))
            };
            let mut ll: Vec<f64> = Vec::new();
            let mut summed: Vec<usize> = Vec::new();
            let mut sdi = (0u32, 0u32);
            if let Some((hr, hp)) = hyps_of {
                let hyps = &analyses[hr].results[hp].hyps;
                ll = vec![0.0; hyps.len()];
                for &(r, p) in &members {
                    let res = &analyses[r].results[p];
                    if res.ll.is_empty() || res.hyps != *hyps {
                        continue;
                    }
                    for (v, x) in ll.iter_mut().zip(&res.ll) {
                        *v += x;
                    }
                    summed.push(r);
                    sdi.0 += res.n_sdi;
                    sdi.1 += res.n_sdi_observed;
                }
            }
            cons.push(Consensus {
                anchor,
                members,
                summed,
                sdi,
                rt: an.apex,
                ll,
                hyps_of,
                best: None,
                min_site_prob: None,
                post: Vec::new(),
                site_prob: Vec::new(),
                status: "",
                delta_loglik: None,
                flr_model: None,
                flr_decoy: None,
            });
        }
    }
    let result_of = |c: &Consensus| {
        let (r, p) = c.hyps_of.unwrap_or(c.anchor);
        &analyses[r].results[p]
    };
    let rep_of = |c: &Consensus| {
        let (r, p) = c.hyps_of.unwrap_or(c.anchor);
        &analyses[r].rows[analyses[r].peaks[p].rep]
    };

    // --- the experiment temperature ---
    let cal: Vec<calibrate::CalPeak> = cons
        .iter()
        .filter(|c| !c.ll.is_empty() && !rep_of(c).decoy)
        .filter(|c| result_of(c).n_real >= 2)
        .map(|c| {
            let res = result_of(c);
            calibrate::CalPeak {
                ll: &c.ll,
                hyps: &res.hyps,
                on_decoy: &res.on_decoy,
                n_slots: rep_of(c).split.n_slots(),
                real_positions: res.positions.len() - res.n_decoy_positions,
                decoy_positions: res.n_decoy_positions,
                sdi_share: f64::from(c.sdi.1) / f64::from(c.sdi.0.max(1)),
            }
        })
        .collect();
    // The prior's centre: the run temperature if the runs' evidence were independent, the run
    // temperature times the number of runs summed if every run repeated the same evidence;
    // the geometric mean of the two, at the mean number of runs summed.
    let n_cal = cons
        .iter()
        .filter(|c| !c.ll.is_empty() && !rep_of(c).decoy && result_of(c).n_real >= 2)
        .map(|c| c.summed.len())
        .collect::<Vec<_>>();
    let mean_runs = if n_cal.is_empty() {
        1.0
    } else {
        n_cal.iter().sum::<usize>() as f64 / n_cal.len() as f64
    };
    let x_prior = temp.value * mean_runs.max(1.0).sqrt();
    let xtemp = match cfg.fixed_experiment_temperature() {
        Some(t) => Temperature::fixed(t),
        None => {
            let f = calibrate::fit(&cal, x_prior);
            Temperature {
                value: f.temperature,
                fit: Some(f),
            }
        }
    };
    drop(cal);

    // --- the calls ---
    for c in cons.iter_mut() {
        let (r, p) = c.hyps_of.unwrap_or(c.anchor);
        let res = &analyses[r].results[p];
        if c.ll.is_empty() {
            // nothing scored: the anchor's own outcome (fixed by the sequence, too many
            // placements, or no spectra)
            let a = &analyses[c.anchor.0].results[c.anchor.1];
            c.status = a.status;
            c.best = a.best;
            c.min_site_prob = a.min_site_prob;
            c.post = a.post.clone();
            c.site_prob = a.site_prob.clone();
            continue;
        }
        let n_slots = analyses[r].rows[analyses[r].peaks[p].rep].split.n_slots();
        let call = model::call(&c.ll, &res.hyps, n_slots, xtemp.value);
        c.delta_loglik = Some(model::delta_loglik(&c.ll, call.best));
        c.status = if res.on_decoy[call.best] {
            status::DECOY_RESIDUE
        } else if call.min_site_prob >= cfg.threshold {
            status::LOCALISED
        } else {
            status::AMBIGUOUS
        };
        c.best = Some(call.best);
        c.min_site_prob = Some(call.min_site_prob);
        c.post = call.post;
        c.site_prob = call.site_prob;
    }
    // --- false localisation rates over the target consensus peaks with a real choice ---
    let flr_idx: Vec<usize> = (0..cons.len())
        .filter(|&i| {
            let c = &cons[i];
            !c.ll.is_empty() && !rep_of(c).decoy && result_of(c).n_real >= 2
        })
        .collect();
    let (rp, dp) = flr_idx.iter().fold((0usize, 0usize), |acc, &i| {
        let res = result_of(&cons[i]);
        (
            acc.0 + res.positions.len() - res.n_decoy_positions,
            acc.1 + res.n_decoy_positions,
        )
    });
    let ratio = if dp > 0 {
        rp as f64 / dp as f64
    } else {
        f64::NAN
    };
    let flr = flr_curve(
        &flr_idx
            .iter()
            .map(|&i| {
                let c = &cons[i];
                (
                    c.min_site_prob.unwrap_or(0.0),
                    c.status == status::DECOY_RESIDUE,
                )
            })
            .collect::<Vec<_>>(),
        ratio,
    );
    for (k, &i) in flr_idx.iter().enumerate() {
        cons[i].flr_model = Some(flr[k].0);
        cons[i].flr_decoy = flr[k].1;
    }

    // --- the table ---
    let n = cons.len();
    let mut c_mod = Vec::with_capacity(n);
    let mut c_pf = Vec::with_capacity(n);
    let mut c_charge = Vec::with_capacity(n);
    let mut c_label = Vec::with_capacity(n);
    let mut c_protein = Vec::with_capacity(n);
    let mut c_rt = Vec::with_capacity(n);
    let mut c_nruns = Vec::with_capacity(n);
    let mut c_nobs = Vec::with_capacity(n);
    let mut c_runs = Vec::with_capacity(n);
    let mut c_nhyp = Vec::with_capacity(n);
    let mut c_best = Vec::with_capacity(n);
    let mut c_sites = Vec::with_capacity(n);
    let mut c_min = Vec::with_capacity(n);
    let mut c_loc = Vec::with_capacity(n);
    let mut c_ondecoy = Vec::with_capacity(n);
    let mut c_dll = Vec::with_capacity(n);
    let mut c_flrm = Vec::with_capacity(n);
    let mut c_flrd = Vec::with_capacity(n);
    let mut c_status = Vec::with_capacity(n);
    let mut c_hyps = Vec::with_capacity(n);
    let mut c_ll = Vec::with_capacity(n);
    for c in &cons {
        let (ar, ap) = c.anchor;
        let anchor_rep = &analyses[ar].rows[analyses[ar].peaks[ap].rep];
        let rep = rep_of(c);
        let res = result_of(c);
        let name = &cfg.mods[rep.modi];
        c_mod.push(name.clone());
        c_pf.push(anchor_rep.pform.clone());
        c_charge.push(anchor_rep.charge);
        c_label.push(if anchor_rep.decoy { "decoy" } else { "target" }.to_string());
        c_protein.push(anchor_rep.protein.clone());
        c_rt.push(c.rt);
        c_nruns.push(c.summed.len() as u32);
        c_nobs.push(c.members.len() as u32);
        c_runs.push(
            c.members
                .iter()
                .map(|&(r, p)| format!("{}:{p}", analyses[r].source))
                .collect::<Vec<_>>()
                .join(";"),
        );
        c_nhyp.push(res.hyps.len() as u32);
        c_min.push(c.min_site_prob);
        c_dll.push(c.delta_loglik);
        c_flrm.push(c.flr_model);
        c_flrd.push(c.flr_decoy);
        c_status.push(c.status.to_string());
        match c.best {
            Some(b) if !c.post.is_empty() => {
                c_best.push(Some(hypotheses::render(
                    &rep.pform,
                    &rep.split,
                    name,
                    &res.hyps[b],
                )));
                c_sites.push(Some(site_string(&rep.split, &res.positions, &c.site_prob)));
                c_loc.push(matches!(c.status, status::LOCALISED | status::UNAMBIGUOUS));
                c_ondecoy.push(res.on_decoy[b]);
            }
            _ => {
                c_best.push(None);
                c_sites.push(None);
                c_loc.push(false);
                c_ondecoy.push(false);
            }
        }
        if c.ll.is_empty() {
            c_hyps.push(None);
            c_ll.push(Vec::new());
        } else {
            c_hyps.push(Some(super::hypothesis_text(&rep.split, &res.hyps)));
            c_ll.push(c.ll.clone());
        }
    }
    let written = write_table_hashed(
        out,
        vec![
            Col::U32("consensus_peak_id".into(), (0..n as u32).collect()),
            Col::Str("localised_mod".into(), c_mod),
            Col::Str("peptidoform".into(), c_pf),
            Col::I32("charge".into(), c_charge),
            Col::Str("label".into(), c_label),
            Col::Str("protein".into(), c_protein),
            Col::F64("consensus_rt".into(), c_rt),
            Col::U32("n_runs".into(), c_nruns),
            Col::U32("n_runs_observed".into(), c_nobs),
            Col::Str("runs".into(), c_runs),
            Col::U32("n_hypotheses".into(), c_nhyp),
            Col::OptStr("best_site_peptidoform".into(), c_best),
            Col::OptStr("site_probabilities".into(), c_sites),
            Col::OptF64("min_site_prob".into(), c_min),
            Col::Bool("localised".into(), c_loc),
            Col::Bool("on_decoy_residue".into(), c_ondecoy),
            Col::OptF64("delta_loglik".into(), c_dll),
            Col::OptF64("site_flr_model".into(), c_flrm),
            Col::OptF64("site_flr_decoy".into(), c_flrd),
            Col::Str("status".into(), c_status),
            Col::OptStr("hypothesis_sites".into(), c_hyps),
            Col::ListF64("hypothesis_loglik".into(), c_ll),
        ],
    )?;
    let with_a_choice = flr_idx.len() as u64;
    let localised = flr_idx
        .iter()
        .filter(|&&i| cons[i].status == status::LOCALISED)
        .count() as u64;
    let at_threshold = flr_idx
        .iter()
        .filter(|&&i| cons[i].min_site_prob.unwrap_or(0.0) >= cfg.threshold)
        .map(|&i| (cons[i].flr_model.unwrap_or(0.0), cons[i].flr_decoy))
        .reduce(|x, y| {
            (
                x.0.max(y.0),
                match (x.1, y.1) {
                    (Some(u), Some(v)) => Some(u.max(v)),
                    (u, v) => u.or(v),
                },
            )
        });
    let runs_summed = |k: usize| {
        flr_idx
            .iter()
            .filter(|&&i| cons[i].summed.len() == k)
            .count()
    };
    let mut by_runs = BTreeMap::new();
    for k in 1..=analyses.len() {
        let v = runs_summed(k);
        if v > 0 {
            by_runs.insert(k.to_string(), v);
        }
    }
    let mut stats = BTreeMap::new();
    stats.insert("consensus_peaks".into(), json!(n));
    stats.insert(
        "target_consensus_peaks".into(),
        json!(cons.iter().filter(|c| !rep_of(c).decoy).count()),
    );
    stats.insert("with_a_choice".into(), json!(with_a_choice));
    stats.insert("localised".into(), json!(localised));
    stats.insert(
        "localisation_rate".into(),
        json!(if with_a_choice > 0 {
            localised as f64 / with_a_choice as f64
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
    stats.insert("with_a_choice_by_runs_summed".into(), json!(by_runs));
    stats.insert("reference_source".into(), json!(reference));
    stats.insert(
        "aligned_sources".into(),
        json!(curves.keys().collect::<Vec<_>>()),
    );
    stats.insert("run_temperature".into(), temp.report());
    stats.insert("experiment_temperature".into(), xtemp.report());
    let elapsed = t0.elapsed().as_millis();
    ArtifactReport {
        logical_name: artifact::LOCALISATION_EXPERIMENT.0.to_string(),
        schema_name: artifact::LOCALISATION_EXPERIMENT.0.to_string(),
        schema_version: artifact::LOCALISATION_EXPERIMENT.1,
        stage: "localise".to_string(),
        rows: written.rows,
        content_hash: written.content_hash.clone(),
        params: json!({
            "config_hash": config_hash,
            "runs": params.iter().map(|p| p.out).collect::<Vec<_>>(),
            "mods": cfg.mods,
            "threshold": cfg.threshold,
            "experiment_temperature": cfg
                .fixed_experiment_temperature()
                .map_or(json!("auto"), |t| json!(t)),
        }),
        stats,
        model_identity: Some("localise-combined-loglik-v1".to_string()),
        elapsed_ms: elapsed,
    }
    .write_for(out)?;
    info!(
        out,
        consensus_peaks = n,
        with_a_choice,
        localised,
        temperature = xtemp.value,
        "localise: experiment-level done"
    );
    Ok(CombineSummary {
        written,
        consensus_peaks: n as u64,
        with_a_choice,
        localised,
        temperature: xtemp.value,
    })
}
