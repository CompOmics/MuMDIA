//! End-to-end tests of the stage on a planted fixture: a scored table and MS2 spectra written
//! to parquet, with
//!
//! - a co-eluting isomer pair (LSDGATPEK, sites S2 and T6) on one peak, the spectra carrying
//!   the ladder of the S2 placement only;
//! - a separated pair (VTDGNSPEK, T2 at 300 s and S6 at 400 s), each apex carrying its own
//!   placement's ladder;
//! - a peptide whose spectra carry the ladder of a placement on the decoy residue
//!   (VADGSPTLK with the site on A2), searched as S5;
//! - a peptide with a single candidate residue (GDPSLK), whose site the sequence fixes;
//! - a decoy and a row above `q_max`.

use super::*;
use mumdia_core::config::{Config, ModPosition};
use mumdia_io::table::write_table;

struct Fixture {
    dir: std::path::PathBuf,
    scored: String,
    spectra: String,
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mumdia_localise_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The z = 1 b/y m/z of `pf` with the localised `name` on `sites`.
fn ladder_of(pf: &str, name: &str, sites: &[usize]) -> Vec<f64> {
    let s = hypotheses::split(pf, name).unwrap();
    ions::build(&s, &[sites.to_vec()], 1, None)
        .ions
        .iter()
        .map(|i| i.mz)
        .collect()
}

/// Rows: (peptidoform, label, base_peptide_id, apex_rt, score, run_psm_q).
type FixtureRow = (&'static str, &'static str, u32, f64, f64, f64);

fn rows() -> Vec<FixtureRow> {
    vec![
        ("LS[Phospho]DGATPEK", "target", 1, 100.0, 0.9, 0.001),
        ("LSDGAT[Phospho]PEK", "target", 1, 100.4, 0.5, 0.002),
        ("VT[Phospho]DGNSPEK", "target", 2, 300.0, 0.8, 0.001),
        ("VTDGNS[Phospho]PEK", "target", 2, 400.0, 0.7, 0.001),
        ("VADGS[Phospho]PTLK", "target", 3, 500.0, 0.6, 0.003),
        ("GDPS[Phospho]LK", "target", 4, 600.0, 0.6, 0.003),
        ("PEPTIDEK", "target", 5, 700.0, 0.9, 0.001),
        ("KEPTAGDS[Phospho]L", "decoy", 1, 150.0, 0.1, 0.5),
        ("KEPNGDTS[Phospho]V", "decoy", 2, 160.0, 0.05, 0.6),
    ]
}

/// Ladders planted at each apex: (apex, peptidoform, true sites).
fn planted() -> Vec<(f64, &'static str, Vec<usize>)> {
    vec![
        (100.0, "LS[Phospho]DGATPEK", vec![1]),
        (300.0, "VT[Phospho]DGNSPEK", vec![1]),
        (400.0, "VT[Phospho]DGNSPEK", vec![5]),
        (500.0, "VADGS[Phospho]PTLK", vec![1]),
        (600.0, "GDPS[Phospho]LK", vec![3]),
    ]
}

fn fixture(name: &str) -> Fixture {
    let planted: Vec<(f64, &str, &str, Vec<usize>)> = planted()
        .into_iter()
        .map(|(apex, pf, sites)| (apex, pf, "Phospho", sites))
        .collect();
    fixture_of(name, &rows(), &planted)
}

/// A scored table of `r` and spectra carrying, at each apex of `planted` (apex, peptidoform,
/// modification, true slots), the ladder of that placement.
fn fixture_of(name: &str, r: &[FixtureRow], planted: &[(f64, &str, &str, Vec<usize>)]) -> Fixture {
    fixture_full(name, r, planted, 0, None)
}

/// [`fixture_of`] for run `source`, with elution bounds `apex -/+ half` on every row when
/// `bounds` is given.
fn fixture_full(
    name: &str,
    r: &[FixtureRow],
    planted: &[(f64, &str, &str, Vec<usize>)],
    source: u32,
    bounds: Option<f64>,
) -> Fixture {
    let dir = scratch(name);
    let scored = dir.join("scored.parquet").to_str().unwrap().to_string();
    let spectra = dir
        .join("spectra_ms2.parquet")
        .to_str()
        .unwrap()
        .to_string();
    let mut cols = vec![
        Col::U32("candidate_id".into(), (0..r.len() as u32).collect()),
        Col::Str(
            "peptidoform".into(),
            r.iter().map(|x| x.0.to_string()).collect(),
        ),
        Col::I32("charge".into(), vec![2; r.len()]),
        Col::Str("label".into(), r.iter().map(|x| x.1.to_string()).collect()),
        Col::Str("protein".into(), vec!["P1".to_string(); r.len()]),
        Col::U32("base_peptide_id".into(), r.iter().map(|x| x.2).collect()),
        Col::F64("apex_rt".into(), r.iter().map(|x| x.3).collect()),
        Col::F64("score".into(), r.iter().map(|x| x.4).collect()),
        Col::F64("run_psm_q".into(), r.iter().map(|x| x.5).collect()),
        Col::U32("source".into(), vec![source; r.len()]),
        Col::I32("selected_peak_rank".into(), vec![0; r.len()]),
    ];
    if let Some(half) = bounds {
        cols.push(Col::F64(
            "elution_lo".into(),
            r.iter().map(|x| x.3 - half).collect(),
        ));
        cols.push(Col::F64(
            "elution_hi".into(),
            r.iter().map(|x| x.3 + half).collect(),
        ));
    }
    write_table(&scored, cols).unwrap();
    // One wide window, a scan a second around every planted apex, each planted ladder
    // present in the five scans nearest its apex, and a fixed scatter of unrelated peaks.
    let ladders: Vec<(f64, Vec<f64>)> = planted
        .iter()
        .map(|(apex, pf, modname, sites)| (*apex, ladder_of(pf, modname, sites)))
        .collect();
    let mut times: Vec<f64> = ladders
        .iter()
        .flat_map(|(apex, _)| (-10i32..=10).map(move |k| apex + k as f64))
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut rt = Vec::new();
    let mut mz: Vec<Vec<f32>> = Vec::new();
    let mut it: Vec<Vec<f32>> = Vec::new();
    for t in times {
        let nearest = ladders
            .iter()
            .map(|(apex, _)| t - apex)
            .min_by(|a, b| a.abs().total_cmp(&b.abs()))
            .unwrap_or(0.0);
        let k = nearest.round() as f32;
        let mut peaks: Vec<(f32, f32)> = (0..40)
            .map(|j| (150.0 + j as f32 * 23.71 + k * 0.013, 50.0))
            .collect();
        for (apex, lad) in &ladders {
            let d = (t - apex).abs();
            if d <= 2.0 + 1e-9 {
                for &m in lad {
                    peaks.push((m as f32, 1000.0 / (1.0 + d as f32)));
                }
            }
        }
        peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
        rt.push(t);
        mz.push(peaks.iter().map(|p| p.0).collect());
        it.push(peaks.iter().map(|p| p.1).collect());
    }
    let ns = rt.len();
    write_table(
        &spectra,
        vec![
            Col::U32("scan_index".into(), (0..ns as u32).collect()),
            Col::Str("id".into(), (0..ns).map(|i| format!("scan={i}")).collect()),
            Col::F64("rt_seconds".into(), rt),
            Col::F64("window_lower".into(), vec![300.0; ns]),
            Col::F64("window_upper".into(), vec![1500.0; ns]),
            Col::F64("window_target".into(), vec![900.0; ns]),
            Col::ListF32("mz".into(), mz),
            Col::ListF32("intensity".into(), it),
        ],
    )
    .unwrap();
    Fixture {
        dir,
        scored,
        spectra,
    }
}

fn rule(residue: char, name: &str, position: ModPosition) -> ResidueMod {
    ResidueMod {
        residue,
        name: name.into(),
        position,
    }
}

fn run_on(f: &Fixture, out: &str, cfg: &LocaliseConfig) -> LocaliseSummary {
    let phospho: Vec<ResidueMod> = "STY"
        .chars()
        .map(|c| rule(c, "Phospho", ModPosition::Anywhere))
        .collect();
    run_with(f, out, cfg, &phospho, None)
}

fn run_with(
    f: &Fixture,
    out: &str,
    cfg: &LocaliseConfig,
    variable_mods: &[ResidueMod],
    fasta: Option<&str>,
) -> LocaliseSummary {
    let base = Config::default();
    run(LocaliseParams {
        scored: &f.scored,
        spectra_ms2: &f.spectra,
        mass_cal: None,
        psms_extracted: None,
        source: None,
        out,
        cfg,
        variable_mods,
        fasta,
        fallback_tol_ppm: base.extract.frag_tol_ppm,
        config_hash: "test",
    })
    .unwrap()
}

struct Out {
    pform: Vec<String>,
    status: Vec<String>,
    peak: Vec<Option<i32>>,
    rep: Vec<bool>,
    best: Vec<String>,
    min_sp: Vec<Option<f64>>,
    own: Vec<Option<f64>>,
    localised: Vec<bool>,
    group_q: Vec<f64>,
    group: Vec<u32>,
    label: Vec<String>,
}

fn read(path: &str) -> Out {
    let t = TableFile::open(path).unwrap();
    Out {
        pform: t.str("peptidoform").unwrap(),
        status: t.str("status").unwrap(),
        peak: t.opt_i32("isomer_peak_id").unwrap(),
        rep: t.bool("is_peak_representative").unwrap(),
        best: t
            .opt_str("best_site_peptidoform")
            .unwrap()
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect(),
        min_sp: t.opt_f64("min_site_prob").unwrap(),
        own: t.opt_f64("peptidoform_probability").unwrap(),
        localised: t.bool("localised").unwrap(),
        group_q: t.f64("isomer_group_q").unwrap(),
        group: t.u32("isomer_group_id").unwrap(),
        label: t.str("label").unwrap(),
    }
}

fn row_of(o: &Out, pf: &str) -> usize {
    o.pform.iter().position(|p| p == pf).unwrap()
}

#[test]
fn planted_isomers_are_localised_and_the_table_is_deterministic() {
    let f = fixture("planted");
    let cfg = LocaliseConfig::default();
    let out = f.dir.join("loc.parquet").to_str().unwrap().to_string();
    let s = run_on(&f, &out, &cfg);
    // the unmodified row is not in the table
    assert_eq!(s.rows, 8);
    let o = read(&out);
    assert!(!o.pform.iter().any(|p| p == "PEPTIDEK"));

    // co-eluting pair: one peak, one representative (the higher score), localised to S2
    let a = row_of(&o, "LS[Phospho]DGATPEK");
    let b = row_of(&o, "LSDGAT[Phospho]PEK");
    assert_eq!(o.status[a], "localised", "{:?}", o.status);
    assert_eq!(o.status[b], "localised");
    assert_eq!(o.best[a], "LS[Phospho]DGATPEK");
    assert_eq!(o.best[b], "LS[Phospho]DGATPEK");
    assert!(o.rep[a] && !o.rep[b]);
    assert_eq!(o.peak[a], o.peak[b]);
    assert!(o.peak[a].is_some());
    assert!(o.min_sp[a].unwrap() >= 0.75);
    assert!(o.own[b].unwrap() < 0.25, "{:?}", o.own[b]);
    assert!(o.localised[a] && o.localised[b]);

    // separated pair: two peaks, each localised to its own planted site
    let c = row_of(&o, "VT[Phospho]DGNSPEK");
    let d = row_of(&o, "VTDGNS[Phospho]PEK");
    assert!(o.rep[c] && o.rep[d]);
    assert_ne!(o.peak[c], o.peak[d]);
    assert_eq!(o.best[c], "VT[Phospho]DGNSPEK");
    assert_eq!(o.best[d], "VTDGNS[Phospho]PEK");
    assert_eq!(o.status[c], "localised");
    assert_eq!(o.status[d], "localised");

    // the decoy residue wins where the planted ladder puts the site on A
    let e = row_of(&o, "VADGS[Phospho]PTLK");
    assert_eq!(o.status[e], "decoy_residue");
    assert_eq!(o.best[e], "VA[Phospho]DGSPTLK");
    assert!(!o.localised[e]);

    // a single candidate residue: fixed by the sequence
    let g = row_of(&o, "GDPS[Phospho]LK");
    assert_eq!(o.status[g], "unambiguous");
    assert!(o.localised[g]);

    // decoys above q_max are not attempted
    for (i, l) in o.label.iter().enumerate() {
        if l == "decoy" {
            assert_eq!(o.status[i], "not_attempted_q");
            assert_eq!(o.peak[i], None);
        }
    }
    // isomer groups: one per (base peptide, composition, label); picked q on one row each
    assert_eq!(o.group[a], o.group[b]);
    assert_eq!(o.group[c], o.group[d]);
    assert_ne!(o.group[a], o.group[c]);
    let winners_a = [a, b].iter().filter(|&&i| o.group_q[i] < 1.0).count();
    assert_eq!(winners_a, 1);

    // deterministic: a second run writes the same bytes
    let out2 = f.dir.join("loc2.parquet").to_str().unwrap().to_string();
    let s2 = run_on(&f, &out2, &cfg);
    assert_eq!(s.written.content_hash, s2.written.content_hash);
    let _ = std::fs::remove_dir_all(&f.dir);
}

/// Without a decoy residue the A placement is not scored, so VADGSPTLK has a choice of S5
/// and T7 only and cannot be called `decoy_residue`.
#[test]
fn no_decoy_residue_scores_no_decoy_placement() {
    let f = fixture("nodecoy");
    let cfg = LocaliseConfig {
        decoy_residue: String::new(),
        ..LocaliseConfig::default()
    };
    let out = f.dir.join("loc.parquet").to_str().unwrap().to_string();
    run_on(&f, &out, &cfg);
    let o = read(&out);
    let e = row_of(&o, "VADGS[Phospho]PTLK");
    assert_ne!(o.status[e], "decoy_residue");
    assert!(!o.best[e].contains("A[Phospho]"));
    let _ = std::fs::remove_dir_all(&f.dir);
}

/// Acetyl with a protein N-terminal rule (residue `*`) and a K rule: the N-terminus is a
/// placement of its own, never a residue `*`, and is a hypothesis only for a peptide that
/// starts a protein (a row carries it there, or the FASTA says so).
///
/// - SAKPEDLTGR, both forms accepted on one peak, spectra carrying the N-terminal ladder:
///   localised to the N-terminus;
/// - GVKDLPETR, only the K form in the table, spectra carrying the K ladder: without the
///   FASTA the N-terminus is no hypothesis and the site is fixed by the sequence; with a
///   FASTA in which it starts a protein after the initiator methionine it is a hypothesis,
///   and the K placement wins it.
#[test]
fn terminal_rules_are_an_n_terminal_placement() {
    let r: Vec<FixtureRow> = vec![
        ("[Acetyl]-SAKPEDLTGR", "target", 1, 100.0, 0.9, 0.001),
        ("SAK[Acetyl]PEDLTGR", "target", 1, 100.3, 0.6, 0.002),
        ("GVK[Acetyl]DLPETR", "target", 2, 300.0, 0.8, 0.001),
    ];
    let planted = vec![
        (100.0, "[Acetyl]-SAKPEDLTGR", "Acetyl", vec![10]),
        (300.0, "GVK[Acetyl]DLPETR", "Acetyl", vec![2]),
    ];
    let f = fixture_of("nterm", &r, &planted);
    let cfg = LocaliseConfig {
        mods: vec!["Acetyl".into()],
        ..LocaliseConfig::default()
    };
    let rules = vec![
        rule('*', "Acetyl", ModPosition::ProteinNTerm),
        rule('K', "Acetyl", ModPosition::Anywhere),
    ];
    let out = f.dir.join("loc.parquet").to_str().unwrap().to_string();
    let s = run_with(&f, &out, &cfg, &rules, None);
    assert_eq!(s.rows, 3, "both Acetyl forms are in the table");
    let o = read(&out);
    let t = TableFile::open(&out).unwrap();
    let sites = t.opt_str("site_probabilities").unwrap();
    let a = row_of(&o, "[Acetyl]-SAKPEDLTGR");
    let b = row_of(&o, "SAK[Acetyl]PEDLTGR");
    assert_eq!(o.peak[a], o.peak[b]);
    assert_eq!(o.status[a], "localised", "{:?}", sites[a]);
    assert_eq!(o.best[a], "[Acetyl]-SAKPEDLTGR");
    assert_eq!(o.best[b], "[Acetyl]-SAKPEDLTGR");
    // the candidate slots: the decoy residue A2, K3 and the N-terminus; no `*`
    let cells: Vec<&str> = sites[a]
        .as_deref()
        .unwrap()
        .split(';')
        .map(|c| c.split(':').next().unwrap())
        .collect();
    assert_eq!(cells, vec!["A2", "K3", "N-term"]);
    let g = row_of(&o, "GVK[Acetyl]DLPETR");
    assert_eq!(o.status[g], "unambiguous");
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}.report.json")).unwrap())
            .unwrap();
    assert_eq!(report["params"]["candidate_residues"][0], "K");
    assert_eq!(report["params"]["n_term"][0], "protein_n_term");

    // with a FASTA in which GVKDLPETR starts a protein after its methionine
    let fasta = f.dir.join("p.fasta").to_str().unwrap().to_string();
    std::fs::write(&fasta, ">P1\nMGVKDLPETRSAKPEDLTGR\n").unwrap();
    let out2 = f.dir.join("loc2.parquet").to_str().unwrap().to_string();
    run_with(&f, &out2, &cfg, &rules, Some(&fasta));
    let o2 = read(&out2);
    let g2 = row_of(&o2, "GVK[Acetyl]DLPETR");
    assert_eq!(o2.status[g2], "localised");
    assert_eq!(o2.best[g2], "GVK[Acetyl]DLPETR");
    // SAKPEDLTGR does not start P1, but its N-terminal form is in the table
    assert_eq!(
        o2.best[row_of(&o2, "SAK[Acetyl]PEDLTGR")],
        "[Acetyl]-SAKPEDLTGR"
    );

    // a terminal rule alone: no residue candidate, the N-terminus only
    let only_term = vec![rule('*', "Acetyl", ModPosition::PeptideNTerm)];
    let out3 = f.dir.join("loc3.parquet").to_str().unwrap().to_string();
    run_with(&f, &out3, &cfg, &only_term, None);
    let rep3: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out3}.report.json")).unwrap())
            .unwrap();
    assert_eq!(rep3["params"]["candidate_residues"][0], "");
    assert_eq!(rep3["params"]["n_term"][0], "peptide_n_term");
    let o3 = read(&out3);
    // no residue rule, so K is no candidate: the peak's placements are the N-terminus and the
    // decoy A2, and the N-terminus is fixed by the sequence
    let a3 = row_of(&o3, "[Acetyl]-SAKPEDLTGR");
    assert_eq!(o3.status[a3], "unambiguous");
    assert_eq!(o3.best[a3], "[Acetyl]-SAKPEDLTGR");
    let _ = std::fs::remove_dir_all(&f.dir);
}

/// Two isomers 9 s apart with overlapping elution bounds: the 5 s rule makes two peaks; the
/// elution-bounds rule makes one when their shared fragments show one peak, and two when the
/// shared-fragment profile dips between the apexes.
#[test]
fn the_elution_bounds_rule_merges_one_peak_and_keeps_two() {
    let r: Vec<FixtureRow> = vec![
        ("VT[Phospho]DGNSPEK", "target", 2, 300.0, 0.8, 0.001),
        ("VTDGNS[Phospho]PEK", "target", 2, 309.0, 0.7, 0.001),
    ];
    let peaks_of = |name: &str, planted: &[(f64, &str, &str, Vec<usize>)], rule: &str| {
        let f = fixture_full(name, &r, planted, 0, Some(10.0));
        let cfg: LocaliseConfig = serde_json::from_value(serde_json::json!({
            "peak_rule": rule
        }))
        .unwrap();
        let out = f.dir.join("loc.parquet").to_str().unwrap().to_string();
        run_on(&f, &out, &cfg);
        let o = read(&out);
        let _ = std::fs::remove_dir_all(&f.dir);
        (o.peak[0], o.peak[1], o.best.clone())
    };
    let one = vec![(300.0, "VT[Phospho]DGNSPEK", "Phospho", vec![1])];
    let (a, b, _) = peaks_of("eb_one_tol", &one, "apex_tolerance");
    assert_ne!(a, b);
    let (a, b, best) = peaks_of("eb_one", &one, "elution_bounds");
    assert_eq!(a, b);
    assert_eq!(best[1], "VT[Phospho]DGNSPEK");
    let two = vec![
        (300.0, "VT[Phospho]DGNSPEK", "Phospho", vec![1]),
        (309.0, "VT[Phospho]DGNSPEK", "Phospho", vec![5]),
    ];
    let (a, b, best) = peaks_of("eb_two", &two, "elution_bounds");
    assert_ne!(a, b);
    assert_eq!(best[0], "VT[Phospho]DGNSPEK");
    assert_eq!(best[1], "VTDGNS[Phospho]PEK");
}

/// Neutral-loss ions and the fitted temperature leave the planted calls standing and are
/// recorded in the report; the fitted temperature is on the grid.
#[test]
fn neutral_loss_and_auto_temperature_run_on_the_fixture() {
    let f = fixture("nl_auto");
    let cfg: LocaliseConfig = serde_json::from_value(serde_json::json!({
        "neutral_loss": true,
        "temperature": "auto"
    }))
    .unwrap();
    assert_eq!(cfg.fixed_temperature(), None);
    let out = f.dir.join("loc.parquet").to_str().unwrap().to_string();
    let s = run_on(&f, &out, &cfg);
    assert!((0.25..=8.0).contains(&s.temperature), "{}", s.temperature);
    let o = read(&out);
    assert_eq!(
        o.best[row_of(&o, "LSDGAT[Phospho]PEK")],
        "LS[Phospho]DGATPEK"
    );
    let rep: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}.report.json")).unwrap())
            .unwrap();
    assert_eq!(rep["params"]["neutral_loss"], true);
    assert_eq!(rep["stats"]["temperature"]["mode"], "auto");
    assert_eq!(rep["schema_version"], 2);
    // the classes of the loss ions have rates of their own
    assert_eq!(
        rep["stats"]["detection_rate_by_class"]
            .as_array()
            .unwrap()
            .len(),
        8
    );
    let _ = std::fs::remove_dir_all(&f.dir);
}

/// Two runs of an experiment: run 1 lacks VADGSPTLK and has the other peptides at the same
/// retention times. The consensus peaks sum both runs' log-likelihoods where both observed
/// the peak (one consensus peak per separated isomer), and only run 0's for VADGSPTLK; the
/// per-run tables are those `run` writes for each run alone at the same temperature.
#[test]
fn the_runs_of_an_experiment_combine_into_consensus_peaks() {
    let planted: Vec<(f64, &str, &str, Vec<usize>)> = planted()
        .into_iter()
        .map(|(apex, pf, sites)| (apex, pf, "Phospho", sites))
        .collect();
    let r0 = rows();
    let r1: Vec<FixtureRow> = rows()
        .into_iter()
        .filter(|x| !x.0.starts_with("VADGS"))
        .collect();
    let f0 = fixture_full("x_r0", &r0, &planted, 0, None);
    let f1 = fixture_full("x_r1", &r1, &planted, 1, None);
    // a fixed run temperature: the per-run tables then equal those of `run` alone
    let cfg = LocaliseConfig {
        temperature: 1.0,
        ..LocaliseConfig::default()
    };
    let phospho: Vec<ResidueMod> = "STY"
        .chars()
        .map(|c| rule(c, "Phospho", ModPosition::Anywhere))
        .collect();
    let base = Config::default();
    let out0 = f0.dir.join("loc.parquet").to_str().unwrap().to_string();
    let out1 = f1.dir.join("loc.parquet").to_str().unwrap().to_string();
    let outx = f0.dir.join("loc_x.parquet").to_str().unwrap().to_string();
    let params = |f: &Fixture, out: &str, s: u32| -> (String, String, String, u32) {
        (f.scored.clone(), f.spectra.clone(), out.to_string(), s)
    };
    let inputs = [params(&f0, &out0, 0), params(&f1, &out1, 1)];
    let s = run_experiment(ExperimentLocaliseParams {
        runs: inputs
            .iter()
            .map(|(scored, spectra, out, src)| LocaliseParams {
                scored,
                spectra_ms2: spectra,
                mass_cal: None,
                psms_extracted: None,
                source: Some(*src),
                out,
                cfg: &cfg,
                variable_mods: &phospho,
                fasta: None,
                fallback_tol_ppm: base.extract.frag_tol_ppm,
                config_hash: "test",
            })
            .collect(),
        out: Some(&outx),
        config_hash: "test",
    })
    .unwrap();
    let c = s.combined.as_ref().unwrap();
    assert!(c.consensus_peaks > 0);
    let x = TableFile::open(&outx).unwrap();
    let pf = x.str("peptidoform").unwrap();
    let n_runs = x.u32("n_runs").unwrap();
    let best = x.opt_str("best_site_peptidoform").unwrap();
    let status = x.str("status").unwrap();
    let label = x.str("label").unwrap();
    let at = |p: &str| pf.iter().position(|q| q == p).unwrap();
    let ls = at("LS[Phospho]DGATPEK");
    assert_eq!(n_runs[ls], 2);
    assert_eq!(status[ls], "localised");
    assert_eq!(best[ls].as_deref(), Some("LS[Phospho]DGATPEK"));
    // the separated pair: two consensus peaks, each from both runs, each its own site
    let vt = at("VT[Phospho]DGNSPEK");
    let vs = at("VTDGNS[Phospho]PEK");
    assert_ne!(vt, vs);
    assert_eq!((n_runs[vt], n_runs[vs]), (2, 2));
    assert_eq!(best[vs].as_deref(), Some("VTDGNS[Phospho]PEK"));
    // observed in run 0 only
    let va = at("VADGS[Phospho]PTLK");
    assert_eq!(n_runs[va], 1);
    assert_eq!(status[va], "decoy_residue");
    // decoys above q_max are no peaks, so no consensus peaks
    assert!(label.iter().all(|l| l == "target"));
    // the per-run tables equal `run` on each run alone at the same (fixed) temperature
    let alone = f1.dir.join("alone.parquet").to_str().unwrap().to_string();
    let a1 = run(LocaliseParams {
        scored: &f1.scored,
        spectra_ms2: &f1.spectra,
        mass_cal: None,
        psms_extracted: None,
        source: Some(1),
        out: &alone,
        cfg: &cfg,
        variable_mods: &phospho,
        fasta: None,
        fallback_tol_ppm: base.extract.frag_tol_ppm,
        config_hash: "test",
    })
    .unwrap();
    assert_eq!(a1.written.content_hash, s.runs[1].written.content_hash);
    // deterministic
    let outx2 = f0.dir.join("loc_x2.parquet").to_str().unwrap().to_string();
    let s2 = run_experiment(ExperimentLocaliseParams {
        runs: inputs
            .iter()
            .map(|(scored, spectra, _, src)| LocaliseParams {
                scored,
                spectra_ms2: spectra,
                mass_cal: None,
                psms_extracted: None,
                source: Some(*src),
                out: if *src == 0 { &out0 } else { &out1 },
                cfg: &cfg,
                variable_mods: &phospho,
                fasta: None,
                fallback_tol_ppm: base.extract.frag_tol_ppm,
                config_hash: "test",
            })
            .collect(),
        out: Some(&outx2),
        config_hash: "test",
    })
    .unwrap();
    assert_eq!(
        s2.combined.unwrap().written.content_hash,
        c.written.content_hash
    );
    let _ = std::fs::remove_dir_all(&f0.dir);
    let _ = std::fs::remove_dir_all(&f1.dir);
}

#[test]
fn flr_curve_is_monotone_and_counts_decoy_wins() {
    // probabilities descending: 0.99 real, 0.95 decoy, 0.9 real, 0.8 real
    let peaks = vec![(0.9, false), (0.99, false), (0.8, false), (0.95, true)];
    let c = flr_curve(&peaks, 3.0);
    // at 0.99: one real, err 0.01
    assert!((c[1].0 - 0.01).abs() < 1e-12);
    assert_eq!(c[1].1, Some(0.0));
    // at 0.8: three real (err 0.01 + 0.1 + 0.2), one decoy x 3 / 3 real
    assert!((c[2].0 - 0.31 / 3.0).abs() < 1e-12);
    assert!((c[2].1.unwrap() - 1.0).abs() < 1e-12);
    // monotone in probability
    let mut by: Vec<(f64, f64)> = peaks.iter().zip(&c).map(|(p, f)| (p.0, f.0)).collect();
    by.sort_by(|a, b| b.0.total_cmp(&a.0));
    for w in by.windows(2) {
        assert!(w[0].1 <= w[1].1 + 1e-15);
    }
    // no decoy positions: no empirical rate
    assert!(flr_curve(&peaks, f64::NAN).iter().all(|x| x.1.is_none()));
}
