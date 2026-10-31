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
use mumdia_core::config::Config;
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

/// The z = 1 b/y m/z of `pf` with the localised phospho on `sites`.
fn ladder(pf: &str, sites: &[usize]) -> Vec<f64> {
    let s = hypotheses::split(pf, "Phospho").unwrap();
    ions::build(&s, &[sites.to_vec()], 1)
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
    let dir = scratch(name);
    let scored = dir.join("scored.parquet").to_str().unwrap().to_string();
    let spectra = dir
        .join("spectra_ms2.parquet")
        .to_str()
        .unwrap()
        .to_string();
    let r = rows();
    write_table(
        &scored,
        vec![
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
            Col::U32("source".into(), vec![0; r.len()]),
            Col::I32("selected_peak_rank".into(), vec![0; r.len()]),
        ],
    )
    .unwrap();
    // One wide window, a scan a second around every planted apex, the planted ladder
    // present in the five scans nearest it, and a fixed scatter of unrelated peaks.
    let mut rt = Vec::new();
    let mut mz: Vec<Vec<f32>> = Vec::new();
    let mut it: Vec<Vec<f32>> = Vec::new();
    for (apex, pf, sites) in planted() {
        let lad = ladder(pf, &sites);
        for k in -10i32..=10 {
            let t = apex + k as f64;
            let mut peaks: Vec<(f32, f32)> = (0..40)
                .map(|j| (150.0 + j as f32 * 23.71 + k as f32 * 0.013, 50.0))
                .collect();
            if k.abs() <= 2 {
                for &m in &lad {
                    peaks.push((m as f32, 1000.0 / (1.0 + k.abs() as f32)));
                }
            }
            peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
            rt.push(t);
            mz.push(peaks.iter().map(|p| p.0).collect());
            it.push(peaks.iter().map(|p| p.1).collect());
        }
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

fn run_on(f: &Fixture, out: &str, cfg: &LocaliseConfig) -> LocaliseSummary {
    let base = Config::default();
    run(LocaliseParams {
        scored: &f.scored,
        spectra_ms2: &f.spectra,
        mass_cal: None,
        psms_extracted: None,
        source: None,
        out,
        cfg,
        variable_mods: &[
            ResidueMod {
                residue: 'S',
                name: "Phospho".into(),
            },
            ResidueMod {
                residue: 'T',
                name: "Phospho".into(),
            },
            ResidueMod {
                residue: 'Y',
                name: "Phospho".into(),
            },
        ],
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
