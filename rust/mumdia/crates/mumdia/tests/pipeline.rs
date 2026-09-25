//! Standalone + determinism tests (docs/14_build_test_deploy_gotchas.md).
//! Craft a tiny library and MS2 set by hand, then drive the extract -> features
//! -> compete -> rescore chain directly on files, asserting the planted target
//! is recovered and the output is reproducible.

use mumdia::spectra::Ms1Scan;
use mumdia::stages;
use mumdia_core::config::Config;
use mumdia_core::types::Ms2Scan;
use mumdia_io::table::{write_table, Col, Table, TableFile};

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
    craft_library_inner(false)
}

/// `share_fragment`: give the decoy the target's `y3` m/z instead of its own, so the two
/// candidates claim the SAME observed peak. Only the peak-claim tests want this; the
/// default library keeps the two candidates disjoint, which is what "a decoy with no
/// matching peaks" above means.
fn craft_library_inner(share_fragment: bool) -> (String, String) {
    let decoy_y3 = if share_fragment { 300.2 } else { 350.8 };
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
            Col::F64(
                "mz".into(),
                vec![200.1, 300.2, 400.3, 250.7, decoy_y3, 450.9],
            ),
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
    // Written in DESCENDING retention time. Every consumer reads these scans through
    // `spectra::load_ms2`, which sorts by `rt_seconds` before returning, so the decoded
    // order is the ascending one either way and no assertion in this file depends on the
    // file order. Writing them out of order is what makes that sort load-bearing: with an
    // already-ascending fixture a stage that re-sorted, filtered or reversed the scans
    // would be indistinguishable from one that did not, and the grouped search now lends
    // ONE decoded buffer to every band, so scan order is a cross-band invariant rather
    // than a per-stage detail.
    let rev = |v: Vec<Vec<f32>>| v.into_iter().rev().collect::<Vec<_>>();
    write_table(
        &path,
        vec![
            Col::U32(
                "scan_index".into(),
                scan_index.into_iter().rev().collect::<Vec<_>>(),
            ),
            Col::Str("id".into(), id.into_iter().rev().collect::<Vec<_>>()),
            Col::F64(
                "rt_seconds".into(),
                rt.into_iter().rev().collect::<Vec<_>>(),
            ),
            Col::U32("window_id".into(), win_id),
            Col::F64("window_target".into(), target),
            Col::F64("window_lower".into(), lower),
            Col::F64("window_upper".into(), upper),
            Col::OptF64("precursor_mz".into(), pmz),
            Col::OptI32("precursor_charge".into(), pz),
            Col::ListF32("mz".into(), rev(mz)),
            Col::ListF32("intensity".into(), rev(inten)),
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
        precursor_span: None,
        fragment_offset: None,
        rt_windows: None,
        sibling_bands: 1,
        scans: None,
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

    let mut cfg = Config::default();
    // `features.emit_pin` is opt-in now: no MuMDIA stage reads the file (rescore builds
    // its own PIN) and it is a ~5.4 GB text write per run on a real library. This test
    // asserts the PIN's header format, so it asks for the artifact explicitly rather
    // than depending on a default whose point is that it costs nothing when unused.
    cfg.features.emit_pin = true;
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
        features_hash: None,
    })
    .unwrap();

    let scored = tmp("scored.parquet");
    stages::rescore::run(stages::rescore::RescoreParams {
        competed: &[competed],
        sources: None,
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

/// `w` is what an orchestrator records in the manifest for `path` without reading the
/// file again (docs/03 "Each artifact is hashed once"), so it must be the hash of the file
/// at that path and the hash and row count its report records.
fn assert_written(w: &mumdia_io::report::Written, path: &str) {
    assert_eq!(
        w.content_hash,
        mumdia_io::hash::blake3_file(path).unwrap(),
        "{path}"
    );
    let rep: mumdia_io::report::ArtifactReport =
        mumdia_io::json::read_json(&format!("{path}.report.json")).unwrap();
    assert_eq!(w.content_hash, rep.content_hash, "{path}");
    assert_eq!(w.rows, rep.rows, "{path}");
}

/// The library-build stages, digest to predict-frag with the native predictors. The
/// predict-frag pair is returned as (precursors, fragments); the two files differ, so a
/// swap in that order fails here rather than only in the smoke run.
#[test]
fn library_build_stages_return_the_hashes_of_the_files_they_wrote() {
    let fasta = tmp("tiny.fasta");
    std::fs::write(
        &fasta,
        ">sp|P1|PROT1\nMPEPTIDEKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCK\n\
         >sp|P2|PROT2\nMAGVLTDLQKRSSLLNELSASSGYRK\n",
    )
    .unwrap();
    let cfg = Config::default();
    let peptides = tmp("peptides.parquet");
    let wd = stages::digest::run_hashed(stages::digest::DigestParams {
        fasta: &fasta,
        out: &peptides,
        cfg: &cfg.digest,
        rng_seed: cfg.rng_seed,
        config_hash: "test",
    })
    .unwrap();
    assert!(wd.rows > 0);
    assert_written(&wd, &peptides);

    let pforms = tmp("peptidoforms.parquet");
    let wp = stages::peptidoforms::run_hashed(stages::peptidoforms::PeptidoformsParams {
        peptides: &peptides,
        out: &pforms,
        cfg: &cfg.peptidoforms,
        config_hash: "test",
    })
    .unwrap();
    assert!(wp.rows > 0);
    assert_written(&wp, &pforms);

    let prec = tmp("lib_precursors_hashed.parquet");
    let frag = tmp("lib_fragments_hashed.parquet");
    let (wprec, wfrag) =
        stages::predict_frag::run_hashed(stages::predict_frag::PredictFragParams {
            rt_placeholder: false,
            peptidoforms: &pforms,
            out_precursors: &prec,
            out_fragments: &frag,
            cfg: &cfg.predict_frag,
            work_dir: &tmp("predict_frag_work"),
            config_hash: "test",
        })
        .unwrap();
    assert_written(&wprec, &prec);
    assert_written(&wfrag, &frag);
    assert_ne!(wprec.content_hash, wfrag.content_hash);
}

/// The search stages from extract to quant through their `run_hashed` entry points,
/// chained as the orchestrators chain them: compete receives the features hash, and quant
/// writes all three of its tables. Each returned value must describe its own file.
#[test]
fn search_stages_return_the_hashes_of_the_files_they_wrote() {
    let (prec, frag) = craft_library();
    let ms2 = craft_ms2_with_decoy(true);
    let win = craft_windows();
    let cfg = Config::default();

    let psms = tmp("psms_hashed.parquet");
    let chrom = tmp("chrom_hashed.parquet");
    let (wpsms, wchrom) = stages::extract::run_hashed(stages::extract::ExtractParams {
        precursor_span: None,
        fragment_offset: None,
        sibling_bands: 1,
        rt_windows: None,
        scans: None,
        ms2: &ms2,
        library_precursors: &prec,
        library_fragments: &frag,
        run_windows: &win,
        ms1: None,
        mass_cal: None,
        out_psms: &psms,
        out_chrom: &chrom,
        restrict_candidates: None,
        cfg: &cfg.extract,
        config_hash: "test",
    })
    .unwrap();
    assert!(wpsms.rows >= 1);
    assert_written(&wpsms, &psms);
    assert_written(&wchrom, &chrom);
    assert_ne!(wpsms.content_hash, wchrom.content_hash);

    let feats = tmp("features_hashed.parquet");
    let wfeat = stages::features::run_hashed(stages::features::FeaturesParams {
        psms: &psms,
        chromatograms: &chrom,
        seed: None,
        out: &feats,
        out_pin: &tmp("hashed.pin"),
        cfg: &cfg.features,
        config_hash: "test",
    })
    .unwrap();
    assert_written(&wfeat, &feats);

    let competed = tmp("competed_hashed.parquet");
    let wcomp = stages::compete::run_hashed(stages::compete::CompeteParams {
        features: &feats,
        out: &competed,
        cfg: &cfg.compete,
        config_hash: "test",
        features_hash: Some(&wfeat.content_hash),
    })
    .unwrap();
    assert_written(&wcomp, &competed);

    let scored = tmp("scored_hashed.parquet");
    let wscored = stages::rescore::run_hashed(stages::rescore::RescoreParams {
        competed: &[competed],
        sources: None,
        out: &scored,
        work_dir: &tmp("rescore_work_hashed"),
        script_dir: "scripts",
        cfg: &cfg.rescore,
        config_hash: "test",
    })
    .unwrap();
    assert_written(&wscored, &scored);

    let qpep = tmp("quant_peptide_hashed.parquet");
    let qprot = tmp("quant_protein_hashed.parquet");
    let qfrag = tmp("quant_fragment_hashed.parquet");
    let wq = stages::quant::run_hashed(stages::quant::QuantParams {
        psms_scored: &scored,
        chromatograms: &chrom,
        out_peptide: &qpep,
        out_protein: &qprot,
        out_fragment: Some(&qfrag),
        out_peak_bounds: None,
        cfg: &cfg.quant,
        config_hash: "test",
    })
    .unwrap();
    assert_written(&wq.peptide, &qpep);
    assert_written(&wq.protein, &qprot);
    assert_written(wq.fragment.as_ref().expect("out_fragment was set"), &qfrag);
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
    // The PIN is opt-in since the parquet handoff became the default; this test compares
    // both artifacts across chunk sizes, so ask for it.
    let mut cfg = Config::default();
    cfg.features.emit_pin = true;
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
    // Every float column bit for bit, so a feature that only differs in the last ulp
    // fails: the f64 bookkeeping and kept feature columns, and the f32 feature columns of
    // the v2 layout (`features::F64_FEATURE_COLUMNS` has the ones that stay f64).
    let (mut n64, mut n32) = (0usize, 0usize);
    for name in a.column_names() {
        if let (Ok(x), Ok(y)) = (a.f64(&name), b.f64(&name)) {
            let xb: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
            let yb: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
            assert_eq!(xb, yb, "column '{name}' differs between chunk sizes");
            n64 += 1;
        } else if let (Ok(x), Ok(y)) = (a.f32(&name), b.f32(&name)) {
            let xb: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
            let yb: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
            assert_eq!(xb, yb, "f32 column '{name}' differs between chunk sizes");
            n32 += 1;
        }
    }
    // The Minimal set: 14 features, of which `charge` and `n_matched_fragments` stay f64,
    // beside the five f64 bookkeeping columns.
    assert_eq!((n64, n32), (7, 12), "f64 and f32 columns compared");
    let schema = mumdia_io::table::TableFile::open(&f_one).unwrap().schema;
    for f in schema.fields() {
        if !stages::features::NON_FEATURE_COLUMNS.contains(&f.name().as_str()) {
            assert_eq!(
                f.data_type(),
                &stages::features::feature_storage_type(f.name()),
                "feature column '{}'",
                f.name()
            );
        }
    }
}

/// A tiny MS1 artifact: two scans bracketing the MS2 retention times, carrying the
/// precursor m/z and its first isotope so the MS1 features have something to read.
fn craft_ms1() -> String {
    let path = tmp("ms1.parquet");
    write_table(
        &path,
        vec![
            Col::U32("scan_index".into(), vec![100, 101]),
            Col::F64("rt_seconds".into(), vec![105.0, 125.0]),
            Col::ListF32(
                "mz".into(),
                vec![
                    vec![499.0, 500.0, 500.5, 501.0],
                    vec![499.0, 500.0, 500.5, 501.0],
                ],
            ),
            Col::ListF32(
                "intensity".into(),
                vec![
                    vec![10.0, 900.0, 400.0, 120.0],
                    vec![12.0, 1100.0, 480.0, 140.0],
                ],
            ),
        ],
    )
    .unwrap();
    path
}

/// A mass-calibration sidecar, as `search-seed` writes it beside its PSM table.
/// `frag_ppm_offset` is what `extract` divides each observed peak m/z by, so two of these
/// give two bands two different views of one shared scan buffer.
fn craft_masscal(ppm_offset: f64, tol_ppm: f64) -> String {
    let path = tmp("masscal.json");
    std::fs::write(
        &path,
        format!("{{\"frag_ppm_offset\": {ppm_offset}, \"frag_tol_ppm\": {tol_ppm}}}"),
    )
    .unwrap();
    path
}

/// Field-by-field equality of two scan lists, including every peak.
///
/// What this does NOT guard is the case it reads like: a stage writing through the shared
/// slice. That cannot compile. `extract` and `search_seed` take `&[Ms2Scan]`, so a stage
/// that wanted to write would have to change its own signature to `&mut`, and the borrow
/// checker rejects the change before any test runs.
///
/// What it does guard is everything the borrow checker does not. That a decode of the same
/// artifact is reproducible, which the grouped path now depends on: `run_groups` decodes
/// the MS2 once for the seeding phase and again for the extraction phase, and the two band
/// tables are joined by candidate id afterwards. That no interior mutability or `unsafe`
/// creeps into `Ms2Scan`, `Peak` or the stages. And, read together with the equality of the
/// artifacts either side of it, that a stage handed the buffer neither consumed nor
/// reordered what the next reader sees.
fn assert_scans_identical(after: &[Ms2Scan], fresh: &[Ms2Scan]) {
    assert_eq!(after.len(), fresh.len(), "scan count changed");
    for (a, b) in after.iter().zip(fresh) {
        assert_eq!(a.scan_index, b.scan_index);
        assert_eq!(a.rt_seconds.to_bits(), b.rt_seconds.to_bits());
        assert_eq!(a.window, b.window);
        assert_eq!(a.peaks, b.peaks, "peaks of scan {} changed", a.scan_index);
    }
}

fn assert_ms1_identical(after: &[Ms1Scan], fresh: &[Ms1Scan]) {
    assert_eq!(after.len(), fresh.len(), "MS1 scan count changed");
    for (a, b) in after.iter().zip(fresh) {
        assert_eq!(a.scan_index, b.scan_index);
        assert_eq!(a.rt_seconds.to_bits(), b.rt_seconds.to_bits());
        assert_eq!(a.mz, b.mz);
        assert_eq!(a.intensity, b.intensity);
    }
}

/// Extract must write the same bytes whether it decodes the spectra itself from `--ms2`
/// and `--ms1` or is handed an already-decoded buffer, and two stages sharing one buffer
/// must not see each other.
///
/// This pins the OLD behaviour: the arm with `scans: None` is exactly what every caller
/// did before the buffer could be lent, and the shared arms have to match it byte for
/// byte. It also pins the read-only premise the sharing rests on, by asserting the lent
/// buffer still equals a freshly decoded one after several extracts have run over it.
///
/// Not `Config::default()`, and not one mass calibration. The default switches off the
/// two mechanisms the read-only argument actually names, so a test run under it asserts
/// nothing about them:
///
/// - `extract.peak_claim` selects the destructive strategies, and the default is `None`.
///   `CoelutionWinner` here runs the two-pass path that rewrites matched intensities, and
///   `craft_library_inner(true)` gives the target and the decoy a fragment in common so
///   the strategy has a peak to take away from one of them. The assertion below that the
///   claim changes the output is what keeps this honest.
/// - `groups.calibration = per_group` is the grouped mode most exposed to a write-back,
///   because each band applies its own `MassOffset` to the SAME shared peaks. Two
///   calibrations are alternated over one buffer here, and each arm has to equal the
///   arm that decoded for itself under the same calibration. Under a write-back of
///   `peak.mz / factor_at(peak.mz)` the second arm would see peaks the first had already
///   shifted, and this is the assertion that would fail.
#[test]
fn extract_from_a_shared_scan_buffer_is_byte_identical_and_read_only() {
    let (prec, frag) = craft_library_inner(true);
    let ms2 = craft_ms2_with_decoy(true);
    let ms1 = craft_ms1();
    let win = craft_windows();
    let mut cfg = Config::default();
    cfg.extract.peak_claim = mumdia_core::config::PeakClaim::CoelutionWinner;
    // Within the 20 ppm tolerance both calibrations write, so band A matches; 40 ppm out
    // of it, so band B matches nothing. Both still walk every peak and divide it by their
    // own factor, which is the operation that must not be written back.
    let cal_a = craft_masscal(-15.0, 20.0);
    let cal_b = craft_masscal(40.0, 20.0);

    let run_one = |shared: Option<stages::extract::SharedScans>,
                   cal: &str,
                   tag: &str|
     -> (String, String, u64) {
        let psms = tmp(&format!("psms_{tag}.parquet"));
        let chrom = tmp(&format!("chrom_{tag}.parquet"));
        let (npsm, _) = stages::extract::run(stages::extract::ExtractParams {
            precursor_span: None,
            fragment_offset: None,
            rt_windows: None,
            sibling_bands: 1,
            scans: shared,
            ms2: &ms2,
            library_precursors: &prec,
            library_fragments: &frag,
            run_windows: &win,
            ms1: Some(&ms1),
            mass_cal: Some(cal),
            out_psms: &psms,
            out_chrom: &chrom,
            restrict_candidates: None,
            cfg: &cfg.extract,
            config_hash: "test",
        })
        .unwrap();
        (psms, chrom, npsm)
    };

    // The old path: extract opens the files itself, once per calibration.
    let (psms_own_a, chrom_own_a, rows_a) = run_one(None, &cal_a, "own_a");
    let (psms_own_b, _chrom_own_b, _) = run_one(None, &cal_b, "own_b");
    assert!(
        rows_a > 0,
        "extract found nothing under the first calibration, so every equality below             compares empty tables"
    );

    let h = |p: &str| mumdia_io::hash::blake3_file(p).unwrap();
    assert_ne!(
        h(&psms_own_a),
        h(&psms_own_b),
        "the two mass calibrations make no difference to the output, so alternating them             over one buffer tests nothing"
    );

    // One decode, lent to four extracts in a row under alternating calibrations, as a
    // grouped run under `calibration = per_group` lends it to its bands.
    let ms2_scans = mumdia::spectra::load_ms2(&ms2).unwrap();
    let ms1_scans = mumdia::spectra::load_ms1(&ms1).unwrap();
    let shared = stages::extract::SharedScans {
        ms2: &ms2_scans,
        ms1: Some(&ms1_scans),
    };
    let (psms_a1, chrom_a1, _) = run_one(Some(shared), &cal_a, "shared_a1");
    let (psms_b1, _chrom_b1, _) = run_one(Some(shared), &cal_b, "shared_b1");
    let (psms_a2, chrom_a2, _) = run_one(Some(shared), &cal_a, "shared_a2");
    let (psms_b2, _chrom_b2, _) = run_one(Some(shared), &cal_b, "shared_b2");

    assert_eq!(
        h(&psms_own_a),
        h(&psms_a1),
        "a lent scan buffer changed psms_extracted"
    );
    assert_eq!(
        h(&psms_own_b),
        h(&psms_b1),
        "a lent scan buffer changed psms_extracted under the second calibration"
    );
    assert_eq!(
        h(&psms_a1),
        h(&psms_a2),
        "the band after a differently calibrated band saw a used buffer"
    );
    assert_eq!(
        h(&psms_b1),
        h(&psms_b2),
        "the band after a differently calibrated band saw a used buffer"
    );
    assert_eq!(
        h(&chrom_own_a),
        h(&chrom_a1),
        "a lent scan buffer changed the chromatograms"
    );
    assert_eq!(h(&chrom_a1), h(&chrom_a2));

    assert_scans_identical(&ms2_scans, &mumdia::spectra::load_ms2(&ms2).unwrap());
    assert_ms1_identical(&ms1_scans, &mumdia::spectra::load_ms1(&ms1).unwrap());

    // Guard: the destructive peak-claim strategy has to be doing something, or the half
    // of this test that covers it is decoration. Same inputs, claiming switched off.
    let mut plain = cfg.clone();
    plain.extract.peak_claim = mumdia_core::config::PeakClaim::None;
    let psms_noclaim = tmp("psms_noclaim.parquet");
    let chrom_noclaim = tmp("chrom_noclaim.parquet");
    stages::extract::run(stages::extract::ExtractParams {
        precursor_span: None,
        fragment_offset: None,
        rt_windows: None,
        sibling_bands: 1,
        scans: None,
        ms2: &ms2,
        library_precursors: &prec,
        library_fragments: &frag,
        run_windows: &win,
        ms1: Some(&ms1),
        mass_cal: Some(&cal_a),
        out_psms: &psms_noclaim,
        out_chrom: &chrom_noclaim,
        restrict_candidates: None,
        cfg: &plain.extract,
        config_hash: "test",
    })
    .unwrap();
    assert_ne!(
        h(&psms_own_a),
        h(&psms_noclaim),
        "the peak-claim strategy changes nothing on this fixture, so the destructive path             is untested here"
    );

    // Guard: the MS1 fixture has to reach the output, or the shared-MS1 half of this test
    // would pass over a buffer nothing reads. Same extract without the MS1 artifact.
    let psms_no_ms1 = tmp("psms_no_ms1.parquet");
    let chrom_no_ms1 = tmp("chrom_no_ms1.parquet");
    stages::extract::run(stages::extract::ExtractParams {
        precursor_span: None,
        fragment_offset: None,
        rt_windows: None,
        sibling_bands: 1,
        scans: None,
        ms2: &ms2,
        library_precursors: &prec,
        library_fragments: &frag,
        run_windows: &win,
        ms1: None,
        mass_cal: Some(&cal_a),
        out_psms: &psms_no_ms1,
        out_chrom: &chrom_no_ms1,
        restrict_candidates: None,
        cfg: &cfg.extract,
        config_hash: "test",
    })
    .unwrap();
    assert_ne!(
        h(&psms_own_a),
        h(&psms_no_ms1),
        "the MS1 fixture does not reach psms_extracted, so sharing it is untested here"
    );
}

/// An EMPTY lent MS1 slice must not be read as "this run has no MS1".
///
/// `SharedScans { ms2, ms1: Some(&[]) }` with `ms1: Some(path)` is a caller bug -- a refactor
/// that builds the struct before the MS1 is decoded, or a caller that wants only the MS2
/// saving -- and before the guard it cost every MS1 feature and every `ms1_mono` /
/// `ms1_iso1` / `ms1_iso2` chromatogram row, silently, because both are written under
/// `!ms1_scans.is_empty()`. Extract now decodes the named artifact instead, so the output
/// is the one the path arm produces.
#[test]
fn extract_does_not_believe_an_empty_lent_ms1_over_a_named_one() {
    let (prec, frag) = craft_library();
    let ms2 = craft_ms2();
    let ms1 = craft_ms1();
    let win = craft_windows();
    let cfg = Config::default();

    let run_one = |shared: Option<stages::extract::SharedScans>, tag: &str| -> (String, String) {
        let psms = tmp(&format!("psms_{tag}.parquet"));
        let chrom = tmp(&format!("chrom_{tag}.parquet"));
        stages::extract::run(stages::extract::ExtractParams {
            precursor_span: None,
            fragment_offset: None,
            rt_windows: None,
            sibling_bands: 1,
            scans: shared,
            ms2: &ms2,
            library_precursors: &prec,
            library_fragments: &frag,
            run_windows: &win,
            ms1: Some(&ms1),
            mass_cal: None,
            out_psms: &psms,
            out_chrom: &chrom,
            restrict_candidates: None,
            cfg: &cfg.extract,
            config_hash: "test",
        })
        .unwrap();
        (psms, chrom)
    };

    let (psms_own, chrom_own) = run_one(None, "ms1_own");
    let ms2_scans = mumdia::spectra::load_ms2(&ms2).unwrap();
    let (psms_lent, chrom_lent) = run_one(
        Some(stages::extract::SharedScans {
            ms2: &ms2_scans,
            ms1: Some(&[]),
        }),
        "ms1_lent_empty",
    );
    let h = |p: &str| mumdia_io::hash::blake3_file(p).unwrap();
    assert_eq!(
        h(&psms_own),
        h(&psms_lent),
        "an empty lent MS1 slice dropped MS1 evidence from psms_extracted"
    );
    assert_eq!(
        h(&chrom_own),
        h(&chrom_lent),
        "an empty lent MS1 slice dropped the MS1 chromatogram rows"
    );

    // The same rule for MS2: an empty lent slice means the caller had nothing to lend,
    // never that the run is empty.
    let (psms_no_ms2, chrom_no_ms2) = run_one(
        Some(stages::extract::SharedScans {
            ms2: &[],
            ms1: Some(&[]),
        }),
        "ms2_lent_empty",
    );
    assert_eq!(h(&psms_own), h(&psms_no_ms2));
    assert_eq!(h(&chrom_own), h(&chrom_no_ms2));
}

/// The same contract for the seed search, whose artifacts are the PSM table and the mass
/// calibration beside it. `<out>.report.json` is excluded on purpose: it records the
/// stage's wall clock.
///
/// `top_n_peaks` is lowered below the fixture's peak count on purpose. The default is 300
/// and these scans carry 8, so under the default `select_peaks` returns every index and
/// its capping branch -- the one that would be rewritten to truncate `scan.peaks` in
/// place, and would then hand the next band and extract capped spectra -- never runs at
/// all.
#[test]
fn search_seed_from_a_shared_scan_buffer_is_byte_identical_and_read_only() {
    let (prec, frag) = craft_library();
    let ms2 = craft_ms2_with_decoy(true);
    let mut cfg = Config::default();
    cfg.search_seed.top_n_peaks = 3;
    // The default is 4, and these candidates have three fragments each, so under the
    // default EVERY arm of this test wrote a zero-row seed table and the equality
    // assertions below compared two empty files. Two is what makes the fixture produce
    // seed rows at all, and is also what lets the capped and uncapped arms differ.
    cfg.search_seed.min_matched_peaks = 2;

    let run_one = |shared: Option<&[Ms2Scan]>, tag: &str| -> (String, u64) {
        let out = tmp(&format!("seed_{tag}.parquet"));
        let rows = stages::search_seed::run(stages::search_seed::SearchSeedParams {
            precursor_span: None,
            fragment_offset: None,
            ms2_scans: shared,
            emit_calibrants: false,
            library: None,
            ms2: &ms2,
            library_precursors: &prec,
            library_fragments: &frag,
            out: &out,
            cfg: &cfg.search_seed,
            bucket_size: cfg.extract.bucket_size,
            config_hash: "test",
        })
        .unwrap();
        (out, rows)
    };

    let (own, own_rows) = run_one(None, "own");
    assert!(
        own_rows > 0,
        "the seed found nothing on this fixture, so every equality below compares empty             tables"
    );
    let scans = mumdia::spectra::load_ms2(&ms2).unwrap();
    let (a, _) = run_one(Some(&scans), "shared_a");
    let (b, _) = run_one(Some(&scans), "shared_b");

    let h = |p: &str| mumdia_io::hash::blake3_file(p).unwrap();
    assert_eq!(h(&own), h(&a), "a lent scan buffer changed seed_psms");
    assert_eq!(h(&a), h(&b), "the second band saw a used buffer");
    assert_eq!(
        std::fs::read(format!("{own}.masscal.json")).unwrap(),
        std::fs::read(format!("{a}.masscal.json")).unwrap(),
        "a lent scan buffer changed the mass calibration"
    );
    assert_eq!(
        std::fs::read(format!("{a}.masscal.json")).unwrap(),
        std::fs::read(format!("{b}.masscal.json")).unwrap()
    );

    assert_scans_identical(&scans, &mumdia::spectra::load_ms2(&ms2).unwrap());

    // Guard: the cap has to bind, or the capping branch this test exists to cover is
    // never entered. Same seed with the cap lifted.
    let mut uncapped = cfg.clone();
    uncapped.search_seed.top_n_peaks = 0;
    let out = tmp("seed_uncapped.parquet");
    stages::search_seed::run(stages::search_seed::SearchSeedParams {
        precursor_span: None,
        fragment_offset: None,
        ms2_scans: None,
        emit_calibrants: false,
        library: None,
        ms2: &ms2,
        library_precursors: &prec,
        library_fragments: &frag,
        out: &out,
        cfg: &uncapped.search_seed,
        bucket_size: uncapped.extract.bucket_size,
        config_hash: "test",
    })
    .unwrap();
    assert_ne!(
        h(&own),
        h(&out),
        "search_seed.top_n_peaks does not bind on this fixture, so select_peaks never             caps and the capping branch is untested here"
    );

    // An empty lent slice is the caller having nothing to lend, not an empty run.
    let (empty, _) = run_one(Some(&[]), "lent_empty");
    assert_eq!(
        h(&own),
        h(&empty),
        "an empty lent MS2 slice was searched instead of the artifact"
    );
}

/// Two isolation windows, one library band under each, built so that the bands' own q
/// scales disagree with the pooled one in the direction the fixed calibrant prefix got
/// wrong (docs/33 section 4a).
///
/// Band A (window 400-500) holds 5,000 clean, high-scoring targets and five decoys that
/// score below everything else in the run (a library must hold both labels). Band B
/// (window 500-600) holds 2,500 targets and 60 decoys whose scores sit among its lowest
/// 1,000 targets, so band B's own q puts those targets above 1% while the pooled q, whose
/// denominator also counts band A, accepts every target of both bands (61 / 7,500). The
/// deviations of band B are wider than band A's, so leaving out any of its pooled-accepted
/// calibrants changes the fit. Every candidate has its own scan, carrying its four
/// fragments at small, deterministic ppm deviations; consecutive fragment m/z are 60 ppm
/// apart, so no scan matches a second candidate.
///
/// `overlap`: window A is 400-501 instead of 400-500, so band B's lowest-m/z targets
/// (500.5 to 501) lie in both windows. Band A then loads them too and serves window B for
/// them, as a grouped plan does across a cut between overlapping windows, and both bands
/// hold the same best PSM for each: the case where the pool must take each accepted
/// candidate's deviations from one band only.
struct TwoBandFixture {
    prec: String,
    frag: String,
    ms2: String,
    /// `(first_row, rows)` of each band in the precursor table.
    spans: [(usize, usize); 2],
    band_b_targets: usize,
    /// Rows both bands load (0 without `overlap`).
    shared_rows: usize,
}

fn craft_two_band_fixture(overlap: bool) -> TwoBandFixture {
    let window_a_upper = if overlap { 501.0 } else { 500.0 };
    let (n_at, n_ad, n_bt, n_bd) = (5_000usize, 5usize, 2_500usize, 60usize);
    let n_a = n_at + n_ad;
    let n = n_a + n_bt + n_bd;
    // Row order is precursor m/z order; band B's decoys are interleaved with its targets.
    let mut is_decoy = vec![false; n];
    // Position on the band's own score ladder: 1 is the best, 0 the worst. Band A's
    // decoys get a negative position, which puts them below every row of the run.
    let mut ladder = vec![0.0f64; n];
    for (i, l) in ladder.iter_mut().enumerate().take(n_at) {
        *l = 1.0 - i as f64 / n_at as f64;
    }
    for i in n_at..n_a {
        is_decoy[i] = true;
        ladder[i] = -1.0 - i as f64;
    }
    // Band B: targets on an even ladder, one decoy after every 16th of the bottom 1,000
    // targets, placed just below the target before it.
    let mut k_target = 0usize;
    let mut k_decoy = 0usize;
    for i in n_a..n {
        let in_bottom = k_target >= n_bt - 1_000;
        let decoy_due = in_bottom && k_decoy < n_bd && (k_target - (n_bt - 1_000)) / 16 > k_decoy;
        if (decoy_due || k_target == n_bt) && k_decoy < n_bd {
            is_decoy[i] = true;
            ladder[i] = 1.0 - (k_target as f64 - 0.5) / n_bt as f64;
            k_decoy += 1;
        } else {
            ladder[i] = 1.0 - k_target as f64 / n_bt as f64;
            k_target += 1;
        }
    }
    assert_eq!(
        (k_target, k_decoy),
        (n_bt, n_bd),
        "the fixture places every row"
    );
    let prec_mz: Vec<f64> = (0..n)
        .map(|i| {
            if i < n_a {
                400.5 + 99.0 * i as f64 / n_a as f64
            } else {
                500.5 + 99.0 * (i - n_a) as f64 / (n - n_a) as f64
            }
        })
        .collect();
    let letters = |mut c: usize| {
        let mut s = String::new();
        loop {
            s.push(char::from(b"ACDEFGHILMNPQRSTVWYK"[c % 20]));
            c /= 20;
            if c == 0 {
                break s;
            }
        }
    };
    // Band B rows window A also selects: the first rows of band B, in m/z order.
    let shared_rows = prec_mz[n_a..]
        .iter()
        .take_while(|&&mz| mz <= window_a_upper)
        .count();
    let prec = tmp("two_band_prec.parquet");
    write_table(
        &prec,
        vec![
            Col::U32("candidate_id".into(), (0..n as u32).collect()),
            Col::U32("peptidoform_id".into(), (0..n as u32).collect()),
            Col::U32("base_peptide_id".into(), (0..n as u32).collect()),
            Col::Str(
                "peptidoform".into(),
                (0..n).map(|i| format!("{}K", letters(i))).collect(),
            ),
            Col::I32("charge".into(), vec![2; n]),
            Col::F64("precursor_mz".into(), prec_mz),
            Col::F32("predicted_irt".into(), (0..n).map(|i| i as f32).collect()),
            Col::Str(
                "label".into(),
                is_decoy
                    .iter()
                    .map(|&d| if d { "decoy" } else { "target" }.to_string())
                    .collect(),
            ),
            Col::Str(
                "protein".into(),
                (0..n)
                    .map(|i| {
                        if is_decoy[i] {
                            format!("DECOY_P{i}")
                        } else {
                            format!("P{i}")
                        }
                    })
                    .collect(),
            ),
            Col::I32("n_fragments".into(), vec![4; n]),
        ],
    )
    .unwrap();
    // Geometric spacing, 60 ppm between consecutive fragments: wider than the seed's 20 ppm
    // tolerance and the calibrant collection's 50 ppm window even after a 9 ppm deviation.
    let frag_mz = |i: usize, k: usize| 200.0 * (1.0 + 60e-6f64).powi((i * 4 + k) as i32);
    let frag = tmp("two_band_frag.parquet");
    write_table(
        &frag,
        vec![
            Col::U32(
                "candidate_id".into(),
                (0..n as u32).flat_map(|i| [i; 4]).collect(),
            ),
            Col::F64(
                "mz".into(),
                (0..n)
                    .flat_map(|i| (0..4).map(move |k| frag_mz(i, k)))
                    .collect(),
            ),
            Col::F32("predicted_intensity".into(), vec![1.0; 4 * n]),
            Col::Str(
                "name".into(),
                (0..n)
                    .flat_map(|_| (2..6).map(|o| format!("y{o}")))
                    .collect(),
            ),
            Col::Str("ion_type".into(), vec!["y".to_string(); 4 * n]),
            Col::I32(
                "ordinal".into(),
                (0..n).flat_map(|_| 2..6).collect::<Vec<i32>>(),
            ),
            Col::I32("frag_charge".into(), vec![1; 4 * n]),
        ],
    )
    .unwrap();
    // One scan per candidate. Summed intensity sets the hyperscore: band A far above band
    // B, and inside each band the ladder above.
    let ms2 = tmp("two_band_ms2.parquet");
    let in_a = |i: usize| i < n_a;
    let mut mz_lists = Vec::with_capacity(n);
    let mut int_lists = Vec::with_capacity(n);
    for i in 0..n {
        let spread = if in_a(i) { 0.4 } else { 1.5 };
        let mz: Vec<f32> = (0..4)
            .map(|k| {
                let dev_ppm = (((i * 7 + k * 3) % 13) as f64 - 6.0) * spread;
                (frag_mz(i, k) * (1.0 + dev_ppm * 1e-6)) as f32
            })
            .collect();
        let total = if in_a(i) && is_decoy[i] {
            // Below band B's lowest (1e3): ln(1 + 500 / (1 + 5)) and less.
            500.0 / (1.0 - ladder[i])
        } else if in_a(i) {
            1.0e6 * (1.0 + ladder[i])
        } else {
            1.0e3 * (1.0 + 99.0 * ladder[i])
        };
        mz_lists.push(mz);
        int_lists.push(vec![(total / 4.0) as f32; 4]);
    }
    let (lo, hi): (Vec<f64>, Vec<f64>) = (0..n)
        .map(|i| {
            if in_a(i) {
                (400.0, window_a_upper)
            } else {
                (500.0, 600.0)
            }
        })
        .unzip();
    write_table(
        &ms2,
        vec![
            Col::U32("scan_index".into(), (0..n as u32).collect()),
            Col::Str("id".into(), (0..n).map(|i| format!("scan={i}")).collect()),
            Col::F64(
                "rt_seconds".into(),
                (0..n).map(|i| 10.0 + i as f64 * 0.5).collect(),
            ),
            Col::U32(
                "window_id".into(),
                (0..n).map(|i| u32::from(!in_a(i))).collect(),
            ),
            Col::F64(
                "window_target".into(),
                lo.iter().zip(&hi).map(|(a, b)| 0.5 * (a + b)).collect(),
            ),
            Col::F64("window_lower".into(), lo),
            Col::F64("window_upper".into(), hi),
            Col::OptF64("precursor_mz".into(), vec![None; n]),
            Col::OptI32("precursor_charge".into(), vec![None; n]),
            Col::ListF32("mz".into(), mz_lists),
            Col::ListF32("intensity".into(), int_lists),
        ],
    )
    .unwrap();
    TwoBandFixture {
        prec,
        frag,
        ms2,
        spans: [(0, n_a + shared_rows), (n_a, n - n_a)],
        band_b_targets: n_bt,
        shared_rows,
    }
}

/// The pooled mass calibration of a 2-band search is the unbanded fit (docs/33 section
/// 4a). Until 2026-09-25 each band offered only its 2,000 best targets beyond its own
/// confident set, which is exact at 8 bands and more and short below: here band B's own q
/// rejects its lowest targets and the pooled q accepts all 2,500, so a 2,000-target offer
/// lost the deviations of the rest and the pooled fit came out narrower than the unbanded
/// one, as it did on the HYE Astral benchmark at 2 and 4 bands.
///
/// Run twice: with adjacent windows, and with windows that overlap across the cut, where
/// both bands hold the same PSM of each shared candidate and the pool must count its
/// deviations once, as the unbanded seed does.
#[test]
fn a_two_band_pooled_mass_calibration_equals_the_unbanded_fit() {
    for overlap in [false, true] {
        pooled_mass_calibration_equals_the_unbanded_fit(overlap);
    }
}

fn pooled_mass_calibration_equals_the_unbanded_fit(overlap: bool) {
    let f = craft_two_band_fixture(overlap);
    let arm = if overlap { "overlap" } else { "adjacent" };
    let cfg = Config::default();
    let seed = |prec: &str, offset: Option<u32>, emit: bool, tag: &str| -> String {
        let out = tmp(&format!("two_band_seed_{arm}_{tag}.parquet"));
        stages::search_seed::run(stages::search_seed::SearchSeedParams {
            precursor_span: None,
            fragment_offset: offset,
            ms2_scans: None,
            emit_calibrants: emit,
            library: None,
            ms2: &f.ms2,
            library_precursors: prec,
            library_fragments: &f.frag,
            out: &out,
            cfg: &cfg.search_seed,
            bucket_size: cfg.extract.bucket_size,
            config_hash: "test",
        })
        .unwrap();
        out
    };
    let json = |p: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(format!("{p}.masscal.json")).unwrap())
            .unwrap()
    };
    let unbanded_seed = seed(&f.prec, None, false, "unbanded");
    let unbanded = json(&unbanded_seed);
    assert_eq!(
        mumdia_io::table::nrows(&unbanded_seed).unwrap(),
        7_565,
        "one seed PSM per candidate, as designed"
    );

    let mut bands = Vec::new();
    for (k, &(first, rows)) in f.spans.iter().enumerate() {
        let slice = tmp(&format!("two_band_slice_{arm}_{k}.parquet"));
        mumdia::groups::write_band_slice(&f.prec, first, rows, &slice).unwrap();
        let out = seed(&slice, Some(first as u32), true, &format!("band{k}"));
        bands.push((out, first as u32, rows as u32));
    }
    // The fixture does what it is for: band B's own q rejects hundreds of the targets the
    // pooled q will accept, more than a 2,000-target prefix could make up for.
    let confident = |path: &str| {
        let t = TableFile::open(path).unwrap();
        t.f64("spectrum_q")
            .unwrap()
            .iter()
            .zip(&t.str("label").unwrap())
            .filter(|(q, l)| **q <= 0.01 && *l == "target")
            .count()
    };
    let own_b = confident(&bands[1].0);
    assert!(
        own_b + 300 < f.band_b_targets && own_b < 2_000,
        "band B's own q must reject its low targets, got {own_b} of {} confident",
        f.band_b_targets
    );

    let pooled = tmp(&format!("two_band_pooled_{arm}.parquet"));
    let seeds: Vec<stages::seed_pool::BandSeed> = bands
        .iter()
        .map(|(path, offset, rows)| stages::seed_pool::BandSeed {
            path: path.clone(),
            offset: *offset,
            rows: *rows,
        })
        .collect();
    stages::seed_pool::run(stages::seed_pool::SeedPoolParams {
        seeds: &seeds,
        masscals: &bands
            .iter()
            .map(|b| mumdia::masscal::json_path(&b.0))
            .collect::<Vec<_>>(),
        calibrants: &bands
            .iter()
            .map(|b| mumdia::masscal::calibrants_path(&b.0))
            .collect::<Vec<_>>(),
        cfg: &cfg.search_seed,
        out: &pooled,
    })
    .unwrap();
    let got = json(&pooled);
    assert_eq!(got["masscal_source"], "pooled_deviations", "{arm}");
    if overlap {
        // The arm does what it is for: both bands hold the shared candidates, and each
        // band's sidecar carries their deviations.
        assert!(f.shared_rows >= 10, "{} shared rows", f.shared_rows);
        let shared = f.spans[1].0 as u32..(f.spans[1].0 + f.shared_rows) as u32;
        for (path, _, _) in &bands {
            let c =
                mumdia::masscal::read_calibrants(&mumdia::masscal::calibrants_path(path)).unwrap();
            let n = c
                .candidate_id
                .iter()
                .filter(|cid| shared.contains(cid))
                .count();
            assert_eq!(
                n,
                4 * f.shared_rows,
                "{path}: every shared candidate's four deviations"
            );
        }
    } else {
        assert_eq!(f.shared_rows, 0);
    }
    // The pooled q accepts every target of both bands, which is what the unbanded seed
    // accepts too, so the two fits see the same deviations.
    assert_eq!(confident(&pooled), 5_000 + f.band_b_targets);
    assert_eq!(confident(&unbanded_seed), 5_000 + f.band_b_targets);
    assert_eq!(
        got["n_dev"], unbanded["n_dev"],
        "{arm}: the pooled fit saw a different calibrant set than the unbanded one"
    );
    // The sidecar stores deviations as f32, so the fitted numbers agree to f32 precision
    // rather than bit for bit (docs/33 section 4a: 8.452381550 against 8.452381790).
    for key in ["frag_ppm_offset", "frag_tol_ppm", "ppm_residual_mad"] {
        let (a, b) = (got[key].as_f64().unwrap(), unbanded[key].as_f64().unwrap());
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "{arm} {key}: pooled {a} against unbanded {b}"
        );
    }
}

/// The ungrouped `run` lends the seed's own MS2 decode to extract when no DeepLC step runs
/// between them (`search_seed::run_returning_scans`). The scans the seed hands back must be
/// exactly what `load_ms2` decodes, the seed must hand back nothing it was lent, and an
/// extract over the handed-back scans must write the bytes of an extract that decodes both
/// artifacts itself: with the run's MS1 lent beside them, and with the MS2 lent alone
/// (`ms1: None`, extract decoding the MS1), which is what `run` does.
#[test]
fn the_seed_hands_back_its_decode_and_extract_over_it_is_byte_identical() {
    let (prec, frag) = craft_library();
    let ms2 = craft_ms2_with_decoy(true);
    let ms1 = craft_ms1();
    let win = craft_windows();
    let cfg = Config::default();
    let seed_params = |out: &str, shared: Option<&[Ms2Scan]>| {
        stages::search_seed::run_returning_scans(stages::search_seed::SearchSeedParams {
            precursor_span: None,
            fragment_offset: None,
            ms2_scans: shared,
            emit_calibrants: false,
            library: None,
            ms2: &ms2,
            library_precursors: &prec,
            library_fragments: &frag,
            out,
            cfg: &cfg.search_seed,
            bucket_size: cfg.extract.bucket_size,
            config_hash: "test",
        })
        .unwrap()
    };
    let seed = tmp("seed_handback.parquet");
    let (_, handed) = seed_params(&seed, None);
    let handed = handed.expect("the seed decoded the scans, so it hands them back");
    assert_scans_identical(&handed, &mumdia::spectra::load_ms2(&ms2).unwrap());
    let (_, again) = seed_params(&tmp("seed_handback_lent.parquet"), Some(&handed));
    assert!(again.is_none(), "a lent buffer is not handed back");

    let extract = |shared: Option<stages::extract::SharedScans>, tag: &str| {
        let psms = tmp(&format!("psms_handback_{tag}.parquet"));
        let chrom = tmp(&format!("chrom_handback_{tag}.parquet"));
        stages::extract::run(stages::extract::ExtractParams {
            precursor_span: None,
            fragment_offset: None,
            rt_windows: None,
            sibling_bands: 1,
            scans: shared,
            ms2: &ms2,
            library_precursors: &prec,
            library_fragments: &frag,
            run_windows: &win,
            ms1: Some(&ms1),
            mass_cal: Some(&format!("{seed}.masscal.json")),
            out_psms: &psms,
            out_chrom: &chrom,
            restrict_candidates: None,
            cfg: &cfg.extract,
            config_hash: "test",
        })
        .unwrap();
        (psms, chrom)
    };
    let own = extract(None, "own");
    let ms1_scans = mumdia::spectra::load_ms1(&ms1).unwrap();
    let lent = extract(
        Some(stages::extract::SharedScans {
            ms2: &handed,
            ms1: Some(&ms1_scans),
        }),
        "lent",
    );
    let ms2_only = extract(
        Some(stages::extract::SharedScans {
            ms2: &handed,
            ms1: None,
        }),
        "ms2_only",
    );
    let h = |p: &str| mumdia_io::hash::blake3_file(p).unwrap();
    assert!(
        mumdia_io::table::nrows(&own.0).unwrap() > 0,
        "extract found nothing on this fixture, so the comparison is between empty tables"
    );
    assert_eq!(
        h(&own.0),
        h(&lent.0),
        "psms_extracted differs over the lent decode"
    );
    assert_eq!(
        h(&own.1),
        h(&lent.1),
        "chromatograms differ over the lent decode"
    );
    assert_eq!(
        (h(&own.0), h(&own.1)),
        (h(&ms2_only.0), h(&ms2_only.1)),
        "extract over a lent MS2 alone must decode the MS1 and write the same bytes"
    );
    // Guard: the MS1 has to reach the output, or the MS2-only arm would pass over an MS1
    // decode nothing reads. The same extract over the lent MS2 with no MS1 artifact.
    let no_ms1_psms = tmp("psms_handback_no_ms1.parquet");
    stages::extract::run(stages::extract::ExtractParams {
        precursor_span: None,
        fragment_offset: None,
        rt_windows: None,
        sibling_bands: 1,
        scans: Some(stages::extract::SharedScans {
            ms2: &handed,
            ms1: None,
        }),
        ms2: &ms2,
        library_precursors: &prec,
        library_fragments: &frag,
        run_windows: &win,
        ms1: None,
        mass_cal: Some(&format!("{seed}.masscal.json")),
        out_psms: &no_ms1_psms,
        out_chrom: &tmp("chrom_handback_no_ms1.parquet"),
        restrict_candidates: None,
        cfg: &cfg.extract,
        config_hash: "test",
    })
    .unwrap();
    assert_ne!(
        h(&own.0),
        h(&no_ms1_psms),
        "the MS1 fixture does not reach psms_extracted, so the MS2-only arm is untested"
    );
}

/// `run-experiment` loads one seed library and index and lends it to every run's seed
/// (`search_seed::SeedLibrary`). A seed over the lent library must write the bytes of a
/// seed that loads its own, on both matchers, for several seeds in a row; and a library
/// built for other settings must be refused rather than searched.
#[test]
fn a_lent_seed_library_gives_the_seed_it_would_have_loaded() {
    let (prec, frag) = craft_library();
    let ms2 = craft_ms2_with_decoy(true);
    let h = |p: &str| mumdia_io::hash::blake3_file(p).unwrap();
    for matcher in [
        mumdia_core::config::MatcherKind::Fragindex,
        mumdia_core::config::MatcherKind::Bucketed,
    ] {
        let mut cfg = Config::default();
        cfg.search_seed.min_matched_peaks = 2;
        cfg.search_seed.matcher = matcher;
        let seed = |out: &str, library: Option<&stages::search_seed::SeedLibrary>| {
            stages::search_seed::run(stages::search_seed::SearchSeedParams {
                precursor_span: None,
                fragment_offset: None,
                ms2_scans: None,
                emit_calibrants: false,
                library,
                ms2: &ms2,
                library_precursors: &prec,
                library_fragments: &frag,
                out,
                cfg: &cfg.search_seed,
                bucket_size: cfg.extract.bucket_size,
                config_hash: "test",
            })
        };
        let own = tmp(&format!("seed_lend_own_{matcher:?}.parquet"));
        assert!(
            seed(&own, None).unwrap() > 0,
            "the fixture must produce seed rows"
        );
        let lib = stages::search_seed::SeedLibrary::load(
            &prec,
            &frag,
            None,
            None,
            &cfg.search_seed,
            cfg.extract.bucket_size,
        )
        .unwrap();
        for k in 0..3 {
            let out = tmp(&format!("seed_lend_{matcher:?}_{k}.parquet"));
            seed(&out, Some(&lib)).unwrap();
            assert_eq!(
                h(&own),
                h(&out),
                "{matcher:?} seed {k} over the lent library"
            );
            assert_eq!(
                std::fs::read(format!("{own}.masscal.json")).unwrap(),
                std::fs::read(format!("{out}.masscal.json")).unwrap()
            );
        }
        // Built at another tolerance: refused.
        let mut other = cfg.clone();
        other.search_seed.fragment_tol_ppm += 5.0;
        let wrong = stages::search_seed::SeedLibrary::load(
            &prec,
            &frag,
            None,
            None,
            &other.search_seed,
            cfg.extract.bucket_size,
        )
        .unwrap();
        let err = seed(&tmp("seed_lend_wrong.parquet"), Some(&wrong))
            .unwrap_err()
            .to_string();
        assert!(err.contains("lent seed library"), "{err}");
    }
}

/// T1: the RT windows `rt-im-train` hands an orchestrated extract in memory give the
/// psms_extracted and chromatograms bytes of the extract that reads the run_windows file
/// the same call wrote, and extract takes them without opening that file. Windows fitted
/// on another precursor table, written to another run_windows path, or covering another
/// candidate count are refused, and the file is read instead, with the same bytes.
///
/// Whether extract opened the file is observed directly: for the length of one extract
/// the file holds bytes that are not parquet, so an extract that reads it fails.
#[test]
fn extract_takes_the_handed_rt_windows_only_when_they_are_its_own() {
    use stages::rt_im_train::{run_in_memory, RtImTrainParams, RtWindows};

    /// Run `f` while `path` holds bytes that are not parquet, then put the file back.
    fn without_file<T>(path: &str, f: impl FnOnce() -> T) -> T {
        let bytes = std::fs::read(path).unwrap();
        std::fs::write(path, b"not a parquet file").unwrap();
        let out = f();
        std::fs::write(path, &bytes).unwrap();
        out
    }

    // The planted target (0) and its decoy (1) of `craft_library`, plus a second target (2)
    // on its own base peptide and iRT, whose fragments the spectra do not carry. Two target
    // anchors with distinct iRTs give a calibrated, BOUNDED window, so the arrays handed
    // over are not the unbounded sentinel of a run without anchors.
    let prec = tmp("t1_prec.parquet");
    let frag = tmp("t1_frag.parquet");
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    write_table(
        &prec,
        vec![
            Col::U32("candidate_id".into(), vec![0, 1, 2]),
            Col::U32("peptidoform_id".into(), vec![0, 1, 2]),
            Col::U32("base_peptide_id".into(), vec![0, 0, 1]),
            Col::Str(
                "peptidoform".into(),
                names(&["PEPTIDEK", "EDITPEPK", "ANQTHERK"]),
            ),
            Col::I32("charge".into(), vec![2, 2, 2]),
            Col::F64("precursor_mz".into(), vec![500.0, 500.0, 501.0]),
            Col::F32("predicted_irt".into(), vec![10.0, 10.0, 20.0]),
            Col::Str("label".into(), names(&["target", "decoy", "target"])),
            Col::Str("protein".into(), names(&["P1", "DECOY_P1", "P2"])),
            Col::I32("n_fragments".into(), vec![3, 3, 3]),
        ],
    )
    .unwrap();
    write_table(
        &frag,
        vec![
            Col::U32("candidate_id".into(), vec![0, 0, 0, 1, 1, 1, 2, 2, 2]),
            Col::F64(
                "mz".into(),
                vec![
                    200.1, 300.2, 400.3, 250.7, 350.8, 450.9, 700.1, 710.2, 720.3,
                ],
            ),
            Col::F32(
                "predicted_intensity".into(),
                vec![1.0, 0.8, 0.6, 1.0, 0.8, 0.6, 1.0, 0.8, 0.6],
            ),
            Col::Str(
                "name".into(),
                names(&["b2", "y3", "y4", "b2", "y3", "y4", "b2", "y3", "y4"]),
            ),
            Col::Str(
                "ion_type".into(),
                names(&["b", "y", "y", "b", "y", "y", "b", "y", "y"]),
            ),
            Col::I32("ordinal".into(), vec![2, 3, 4, 2, 3, 4, 2, 3, 4]),
            Col::I32("frag_charge".into(), vec![1; 9]),
        ],
    )
    .unwrap();
    let ms2 = craft_ms2_with_decoy(true);
    let seed = tmp("t1_seed.parquet");
    write_table(
        &seed,
        vec![
            Col::U32("candidate_id".into(), vec![0, 2]),
            Col::U32("base_peptide_id".into(), vec![0, 1]),
            Col::F64("spectrum_q".into(), vec![0.001, 0.001]),
            Col::F64("score".into(), vec![5.0, 4.0]),
            Col::F64("observed_rt".into(), vec![140.0, 240.0]),
            Col::Str("label".into(), names(&["target", "target"])),
        ],
    )
    .unwrap();

    let cfg = Config::default();
    let fit = |library: &str, windows: &str| -> RtWindows {
        let (_, kept) = run_in_memory(RtImTrainParams {
            precursor_span: None,
            seed_psms: &seed,
            library_precursors: library,
            out_windows: windows,
            out_cal: &format!("{windows}.cal.json"),
            cfg: &cfg.rt_im_train,
            config_hash: "test",
            anchor_irt_from_seed: false,
        })
        .unwrap();
        kept.expect("no NaN bound, so the windows are handed over")
    };
    let extract = |rt_windows: Option<RtWindows>,
                   run_windows: &str,
                   tag: &str|
     -> anyhow::Result<(String, String)> {
        let psms = tmp(&format!("t1_psms_{tag}.parquet"));
        let chrom = tmp(&format!("t1_chrom_{tag}.parquet"));
        stages::extract::run(stages::extract::ExtractParams {
            precursor_span: None,
            fragment_offset: None,
            rt_windows,
            sibling_bands: 1,
            scans: None,
            ms2: &ms2,
            library_precursors: &prec,
            library_fragments: &frag,
            run_windows,
            ms1: None,
            mass_cal: None,
            out_psms: &psms,
            out_chrom: &chrom,
            restrict_candidates: None,
            cfg: &cfg.extract,
            config_hash: "test",
        })?;
        Ok((psms, chrom))
    };
    let bytes = |(psms, chrom): &(String, String)| {
        (std::fs::read(psms).unwrap(), std::fs::read(chrom).unwrap())
    };

    let windows = tmp("t1_run_windows.parquet");
    let handed = fit(&prec, &windows);
    let t = Table::read(&windows).unwrap();
    assert!(
        t.f64("rt_lo").unwrap()[0].is_finite() && t.f64("rt_hi").unwrap()[0].is_finite(),
        "the fixture has to fit a bounded window, or the arrays handed over are the sentinel"
    );
    let from_file = extract(None, &windows, "file").unwrap();
    assert!(
        Table::read(&from_file.0)
            .unwrap()
            .u32("candidate_id")
            .unwrap()
            .contains(&0),
        "the planted target must be extracted, or the comparison is of empty tables"
    );
    let from_file = bytes(&from_file);

    // Handed over and taken: the same bytes, and the file was not opened.
    let from_memory = without_file(&windows, || extract(Some(handed), &windows, "memory"))
        .expect("the handed windows are used, so the unreadable file is never opened");
    let from_memory = bytes(&from_memory);
    assert_eq!(
        from_memory.0, from_file.0,
        "psms_extracted, in memory vs file"
    );
    assert_eq!(
        from_memory.1, from_file.1,
        "chromatograms, in memory vs file"
    );

    // Each refusal, built twice: once to show the file is read (it fails while the file is
    // unreadable), once to show that reading it gives the bytes above.
    let prec_copy = tmp("t1_prec_copy.parquet");
    std::fs::copy(&prec, &prec_copy).unwrap();
    let windows_copy = tmp("t1_run_windows_copy.parquet");
    let refusals: [(&str, &dyn Fn() -> RtWindows); 3] = [
        // Fitted on another precursor table holding the same candidates.
        ("library", &|| fit(&prec_copy, &windows)),
        // Written to another run_windows path.
        ("path", &|| fit(&prec, &windows_copy)),
        // Fitted for this library path and this file, over a different candidate count:
        // the table at the path changed after the fit.
        ("count", &|| {
            let (p, w) = (
                std::fs::read(&prec).unwrap(),
                std::fs::read(&windows).unwrap(),
            );
            write_table(
                &prec,
                vec![
                    Col::U32("candidate_id".into(), vec![0, 1]),
                    Col::F32("predicted_irt".into(), vec![10.0, 10.0]),
                ],
            )
            .unwrap();
            let fitted = fit(&prec, &windows);
            std::fs::write(&prec, p).unwrap();
            std::fs::write(&windows, w).unwrap();
            assert_eq!(fitted.len(), 2);
            fitted
        }),
    ];
    for (tag, refused) in refusals {
        let w = refused();
        assert!(
            without_file(&windows, || extract(Some(w), &windows, &format!("{tag}_x"))).is_err(),
            "{tag}: windows that are not this extract's were used instead of the file"
        );
        let got = bytes(&extract(Some(refused()), &windows, tag).unwrap());
        assert_eq!(
            got.0, from_file.0,
            "{tag}: psms_extracted after the refusal"
        );
        assert_eq!(got.1, from_file.1, "{tag}: chromatograms after the refusal");
    }
}
