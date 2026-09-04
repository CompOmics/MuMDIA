//! Standalone + determinism tests (PLAN.md Section 3.5, Section 7 determinism).
//! Craft a tiny library and MS2 set by hand, then drive the extract -> features
//! -> compete -> rescore chain directly on files, asserting the planted target
//! is recovered and the output is reproducible.

use mumdia::stages;
use mumdia_core::config::Config;
use mumdia_io::table::{write_table, Col, Table};

fn tmp(name: &str) -> String {
    // Unique per call: cargo runs tests concurrently in one process, and several
    // tests craft files with the same logical name (ms2.parquet, ...). A shared
    // path made them race (a half-written file read by another test). A
    // per-process dir + atomic counter gives every call its own file.
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!("mumdia_pipeline_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    dir.join(format!("{n}_{name}"))
        .to_str()
        .unwrap()
        .to_string()
}

/// Two candidates in the same isolation window: a target whose three fragments
/// are planted in several consecutive scans, and a decoy with no matching peaks.
fn craft_library() -> (String, String) {
    let prec = tmp("lib_prec.parquet");
    let frag = tmp("lib_frag.parquet");
    write_table(
        &prec,
        vec![
            Col::U32("candidate_id".into(), vec![0, 1]),
            Col::U32("peptidoform_id".into(), vec![0, 1]),
            Col::U32("base_peptide_id".into(), vec![0, 0]),
            Col::Str(
                "peptidoform".into(),
                vec!["PEPTIDEK".into(), "EDITPEPK".into()],
            ),
            Col::I32("charge".into(), vec![2, 2]),
            Col::F64("precursor_mz".into(), vec![500.0, 500.0]),
            Col::F32("predicted_irt".into(), vec![10.0, 10.0]),
            Col::Str("label".into(), vec!["target".into(), "decoy".into()]),
            Col::Str("protein".into(), vec!["P1".into(), "DECOY_P1".into()]),
            Col::I32("n_fragments".into(), vec![3, 3]),
        ],
    )
    .unwrap();
    write_table(
        &frag,
        vec![
            Col::U32("candidate_id".into(), vec![0, 0, 0, 1, 1, 1]),
            Col::F64("mz".into(), vec![200.1, 300.2, 400.3, 250.7, 350.8, 450.9]),
            Col::F32(
                "predicted_intensity".into(),
                vec![1.0, 0.8, 0.6, 1.0, 0.8, 0.6],
            ),
            Col::Str(
                "name".into(),
                vec![
                    "b2".into(),
                    "y3".into(),
                    "y4".into(),
                    "b2".into(),
                    "y3".into(),
                    "y4".into(),
                ],
            ),
            Col::Str(
                "ion_type".into(),
                vec![
                    "b".into(),
                    "y".into(),
                    "y".into(),
                    "b".into(),
                    "y".into(),
                    "y".into(),
                ],
            ),
            Col::I32("ordinal".into(), vec![2, 3, 4, 2, 3, 4]),
            Col::I32("frag_charge".into(), vec![1, 1, 1, 1, 1, 1]),
        ],
    )
    .unwrap();
    (prec, frag)
}

/// MS2 scans: five consecutive scans in one window carrying the target's three
/// fragments; the decoy's fragments never appear.
fn craft_ms2() -> String {
    craft_ms2_with_decoy(false)
}

/// Variant used by the full rescoring test. A valid target-decoy FDR calculation
/// requires at least one extracted example of each label.
fn craft_ms2_with_decoy(include_decoy: bool) -> String {
    let path = tmp("ms2.parquet");
    let n = 5;
    let scan_index: Vec<u32> = (0..n).collect();
    let id: Vec<String> = (0..n).map(|i| format!("scan={i}")).collect();
    let rt: Vec<f64> = (0..n).map(|i| 100.0 + 10.0 * i as f64).collect();
    let win_id = vec![0u32; n as usize];
    let target = vec![500.0; n as usize];
    let lower = vec![498.0; n as usize];
    let upper = vec![502.0; n as usize];
    let pmz: Vec<Option<f64>> = vec![Some(500.0); n as usize];
    let pz: Vec<Option<i32>> = vec![None; n as usize];
    // each scan: target's three fragments plus a couple of noise peaks
    let mz: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            if include_decoy {
                vec![120.0, 200.1, 250.7, 300.2, 350.8, 400.3, 450.9, 600.0]
            } else {
                vec![120.0, 200.1, 300.2, 400.3, 600.0]
            }
        })
        .collect();
    let inten: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let a = 1000.0 + 100.0 * i as f32;
            if include_decoy {
                vec![50.0, a, a * 0.7, a * 0.8, a * 0.56, a * 0.6, a * 0.42, 40.0]
            } else {
                vec![50.0, a, a * 0.8, a * 0.6, 40.0]
            }
        })
        .collect();
    write_table(
        &path,
        vec![
            Col::U32("scan_index".into(), scan_index),
            Col::Str("id".into(), id),
            Col::F64("rt_seconds".into(), rt),
            Col::U32("window_id".into(), win_id),
            Col::F64("window_target".into(), target),
            Col::F64("window_lower".into(), lower),
            Col::F64("window_upper".into(), upper),
            Col::OptF64("precursor_mz".into(), pmz),
            Col::OptI32("precursor_charge".into(), pz),
            Col::ListF32("mz".into(), mz),
            Col::ListF32("intensity".into(), inten),
        ],
    )
    .unwrap();
    path
}

fn craft_windows() -> String {
    let path = tmp("windows.parquet");
    write_table(
        &path,
        vec![
            Col::U32("candidate_id".into(), vec![0, 1]),
            Col::F64("rt_pred_cal".into(), vec![120.0, 120.0]),
            Col::F64("rt_lo".into(), vec![90.0, 90.0]),
            Col::F64("rt_hi".into(), vec![150.0, 150.0]),
            Col::OptF64("im_pred_cal".into(), vec![None, None]),
            Col::OptF64("im_lo".into(), vec![None, None]),
            Col::OptF64("im_hi".into(), vec![None, None]),
        ],
    )
    .unwrap();
    path
}

fn run_extract(prec: &str, frag: &str, ms2: &str, win: &str, tag: &str) -> (String, String) {
    let cfg = Config::default();
    let psms = tmp(&format!("psms_{tag}.parquet"));
    let chrom = tmp(&format!("chrom_{tag}.parquet"));
    stages::extract::run(stages::extract::ExtractParams {
        ms2,
        library_precursors: prec,
        library_fragments: frag,
        run_windows: win,
        ms1: None,
        mass_cal: None,
        out_psms: &psms,
        out_chrom: &chrom,
        restrict_candidates: None,
        cfg: &cfg.extract,
        config_hash: "test",
    })
    .unwrap();
    (psms, chrom)
}

#[test]
fn extract_recovers_planted_target_and_is_deterministic() {
    let (prec, frag) = craft_library();
    let ms2 = craft_ms2();
    let win = craft_windows();

    let (psms1, _c1) = run_extract(&prec, &frag, &ms2, &win, "a");
    let (psms2, _c2) = run_extract(&prec, &frag, &ms2, &win, "b");

    let t1 = Table::read(&psms1).unwrap();
    let t2 = Table::read(&psms2).unwrap();
    // The target (candidate 0) is accepted; the decoy (1) is not.
    let cids1 = t1.u32("candidate_id").unwrap();
    assert!(
        cids1.contains(&0),
        "target candidate not extracted: {cids1:?}"
    );
    assert!(!cids1.contains(&1), "decoy should not be extracted");
    // Deterministic: same rows and apex across two runs.
    assert_eq!(
        t1.u32("candidate_id").unwrap(),
        t2.u32("candidate_id").unwrap()
    );
    assert_eq!(t1.f64("apex_rt").unwrap(), t2.f64("apex_rt").unwrap());
    // Apex should be the last (most intense) scan at rt 140.
    let apex = t1.f64("apex_rt").unwrap();
    assert_eq!(apex[0], 140.0);
}

#[test]
fn features_compete_rescore_run_on_crafted_input() {
    let (prec, frag) = craft_library();
    let ms2 = craft_ms2_with_decoy(true);
    let win = craft_windows();
    let (psms, chrom) = run_extract(&prec, &frag, &ms2, &win, "frc");

    let cfg = Config::default();
    let feats = tmp("features.parquet");
    let pin = tmp("run.pin");
    stages::features::run(stages::features::FeaturesParams {
        psms: &psms,
        chromatograms: &chrom,
        seed: None,
        out: &feats,
        out_pin: &pin,
        cfg: &cfg.features,
        config_hash: "test",
    })
    .unwrap();
    // PIN header exists.
    let pin_text = std::fs::read_to_string(&pin).unwrap();
    assert!(pin_text.starts_with("SpecId\tLabel\tScanNr"));

    let competed = tmp("competed.parquet");
    stages::compete::run(stages::compete::CompeteParams {
        features: &feats,
        out: &competed,
        cfg: &cfg.compete,
        config_hash: "test",
    })
    .unwrap();

    let scored = tmp("scored.parquet");
    stages::rescore::run(stages::rescore::RescoreParams {
        competed: &[competed],
        out: &scored,
        work_dir: &tmp("rescore_work"),
        script_dir: "scripts",
        cfg: &cfg.rescore,
        config_hash: "test",
    })
    .unwrap();
    let t = Table::read(&scored).unwrap();
    assert!(t.nrows >= 1);
    assert!(t.column_names().contains(&"q_value".to_string()));
}

/// Synthetic `psms_extracted` + `chromatograms` pair with enough candidates to span
/// several chunks: varying fragment counts, varying trace lengths, MS1 XIC rows, one
/// never-observed fragment with an empty trace, and one candidate with no chromatogram
/// rows at all.
fn craft_feature_inputs(n_cand: u32) -> (String, String) {
    let psms = tmp("psms_chunk.parquet");
    let chrom = tmp("chrom_chunk.parquet");
    let (mut cid, mut base): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let (mut apex_rt, mut cal, mut mz): (Vec<f64>, Vec<f64>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut apex_int: Vec<f32> = Vec::new();
    let (mut nmatch, mut corun, mut z): (Vec<i32>, Vec<i32>, Vec<i32>) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut label, mut pform, mut prot): (Vec<String>, Vec<String>, Vec<String>) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut ccid, mut cname): (Vec<u32>, Vec<String>) = (Vec::new(), Vec::new());
    let mut cfmz: Vec<f64> = Vec::new();
    let mut cpint: Vec<f32> = Vec::new();
    let (mut crt, mut cint): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
    for c in 0..n_cand {
        let apex = 100.0 + c as f64 * 7.0;
        cid.push(c);
        base.push(c / 2); // two charges of one peptide share a base id
        apex_rt.push(apex);
        cal.push(apex - 1.5);
        mz.push(500.0 + c as f64);
        apex_int.push(1000.0 + c as f32);
        nmatch.push(3);
        corun.push(2);
        z.push(2 + (c % 2) as i32);
        label.push(if c % 3 == 0 { "decoy" } else { "target" }.to_string());
        pform.push(format!("PEPTIDEK{}", c / 2));
        prot.push(format!("P{c};Q{c}"));
        if c % 5 == 4 {
            continue; // candidate with no chromatogram rows
        }
        // Shared window grid for this candidate; length varies with the candidate.
        let npts = 5 + (c % 4) as usize;
        let grid: Vec<f32> = (0..npts).map(|k| (apex - 4.0 + k as f64) as f32).collect();
        let nfrag = 2 + (c % 3);
        for f in 0..nfrag {
            ccid.push(c);
            cname.push(format!("y{}", f + 1));
            cfmz.push(200.0 + f as f64 * 30.0);
            cpint.push(1.0 / (f + 1) as f32);
            if f == nfrag - 1 && c % 4 == 0 {
                crt.push(Vec::new()); // predicted but never observed
                cint.push(Vec::new());
            } else {
                crt.push(grid.clone());
                cint.push(
                    (0..npts)
                        .map(|k| ((k + 1) as f32) * (100.0 - f as f32 * 10.0))
                        .collect(),
                );
            }
        }
        for (iso, nm) in ["ms1_mono", "ms1_iso1", "ms1_iso2"].iter().enumerate() {
            ccid.push(c);
            cname.push(nm.to_string());
            cfmz.push(500.0 + c as f64 + iso as f64 * 0.5);
            cpint.push(0.0);
            crt.push(grid.clone());
            cint.push((0..npts).map(|k| (k as f32 + 1.0) * 50.0).collect());
        }
    }
    let n = cid.len();
    write_table(
        &psms,
        vec![
            Col::U32("candidate_id".into(), cid),
            Col::F64("apex_rt".into(), apex_rt),
            Col::F32("apex_intensity".into(), apex_int),
            Col::I32("n_matched_fragments".into(), nmatch),
            Col::I32("coelution_run".into(), corun),
            Col::F64("rt_pred_cal".into(), cal),
            Col::I32("charge".into(), z),
            Col::Str("label".into(), label),
            Col::U32("base_peptide_id".into(), base),
            Col::Str("peptidoform".into(), pform),
            Col::Str("protein".into(), prot),
            Col::F64("precursor_mz".into(), mz),
            Col::OptF64("ms1_mono".into(), vec![Some(900.0); n]),
            Col::OptF64("ms1_iso1".into(), vec![Some(450.0); n]),
            Col::OptF64("ms1_iso2".into(), vec![Some(120.0); n]),
            Col::OptF64("ms1_isom1".into(), vec![Some(30.0); n]),
        ],
    )
    .unwrap();
    write_table(
        &chrom,
        vec![
            Col::U32("candidate_id".into(), ccid),
            Col::Str("frag_name".into(), cname),
            Col::F64("frag_mz".into(), cfmz),
            Col::F32("predicted_intensity".into(), cpint),
            Col::LargeListF32("rt".into(), crt),
            Col::LargeListF32("intensity".into(), cint),
        ],
    )
    .unwrap();
    (psms, chrom)
}

/// The chunk size is a memory knob, not a semantic one: one chunk for the whole run and
/// one candidate per chunk must produce the same feature values, the same row order and
/// the same PIN bytes. Only the parquet row-group boundaries may differ, so the tables
/// are compared column by column rather than by file hash.
#[test]
fn features_chunking_is_value_preserving() {
    let (psms, chrom) = craft_feature_inputs(23);
    let cfg = Config::default();
    let run = |tag: &str, chunk_rows: usize| -> (String, String) {
        let feats = tmp(&format!("features_{tag}.parquet"));
        let pin = tmp(&format!("features_{tag}.pin"));
        stages::features::run_with_chunk_rows(
            stages::features::FeaturesParams {
                psms: &psms,
                chromatograms: &chrom,
                seed: None,
                out: &feats,
                out_pin: &pin,
                cfg: &cfg.features,
                config_hash: "test",
            },
            chunk_rows,
        )
        .unwrap();
        (feats, pin)
    };
    let (f_one, pin_one) = run("one", 1_000_000);
    let (f_many, pin_many) = run("many", 1);

    assert_eq!(
        std::fs::read_to_string(&pin_one).unwrap(),
        std::fs::read_to_string(&pin_many).unwrap(),
        "PIN bytes differ between chunk sizes"
    );
    let a = Table::read(&f_one).unwrap();
    let b = Table::read(&f_many).unwrap();
    assert_eq!(a.nrows, b.nrows);
    assert!(a.nrows > 1);
    assert_eq!(a.column_names(), b.column_names());
    assert_eq!(
        a.u32("candidate_id").unwrap(),
        b.u32("candidate_id").unwrap()
    );
    assert_eq!(a.str("peptidoform").unwrap(), b.str("peptidoform").unwrap());
    // Every f64 column bit for bit, so a feature that only differs in the last ulp fails.
    for name in a.column_names() {
        if let (Ok(x), Ok(y)) = (a.f64(&name), b.f64(&name)) {
            let xb: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
            let yb: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
            assert_eq!(xb, yb, "column '{name}' differs between chunk sizes");
        }
    }
}
