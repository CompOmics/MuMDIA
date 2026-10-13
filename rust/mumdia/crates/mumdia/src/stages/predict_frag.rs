//! Stage C `mumdia predict-frag`: spectral library generation
//! (docs/06_predict_frag_index_matchers.md). Experiment-wide and run-independent.
//! Computes precursor and b/y fragment m/z from the UniMod-backed mass model,
//! predicted fragment intensities and iRT (native or MS2PIP/DeepLC sidecars over
//! the file contract), keeps top-N fragments, and assigns `candidate_id` as the
//! precursor-m/z sort key so the extractor's index build is a contiguous range
//! (docs/09_extract.md).

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{bail, Result};
use mumdia_core::config::{FragPredictorKind, ImPredictorKind, PredictFragConfig, RtPredictorKind};
use mumdia_core::mass::{parse_peptidoform, Fragment, ParsedPeptidoform};
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table_hashed, Col, TableFile};
use serde_json::json;
use tracing::info;

use crate::predict::{FragmentPredictor, NativeFrag, NativeRt, RtPredictor};
use crate::sidecar;
use rayon::prelude::*;

pub struct PredictFragParams<'a> {
    pub peptidoforms: &'a str,
    pub out_precursors: &'a str,
    pub out_fragments: &'a str,
    pub cfg: &'a PredictFragConfig,
    pub work_dir: &'a str,
    pub config_hash: &'a str,
    /// Write the native retention-time model's iRT in place of DeepLC's, because the caller
    /// will re-predict every row with the multi-head calibration before anything reads it
    /// (`predict_frag.defer_deeplc_to_multihead`). Only an orchestrator that runs that
    /// calibration, and checks afterwards that it re-predicted every row, may set it; the
    /// library's model identity then says the iRT is a placeholder.
    pub rt_placeholder: bool,
}

/// A candidate before intensity/iRT assignment.
struct Raw {
    peptidoform_id: u32,
    base_peptide_id: u32,
    peptidoform: String,
    charge: i32,
    label: String,
    protein: String,
    precursor_mz: f64,
    frags: Vec<Fragment>,
    irt: f32,
    /// Predicted 1/K0 (V s cm^-2); `None` without an ion-mobility predictor.
    im: Option<f64>,
    frag_int: Vec<f32>,
    /// Parsed form kept from phase A so `assign_rt`/`assign_intensities` reuse
    /// it instead of re-parsing `peptidoform` (identical parse output).
    parsed: ParsedPeptidoform,
}

pub fn run(p: PredictFragParams) -> Result<(u64, u64)> {
    run_hashed(p).map(|(prec, frag)| (prec.rows, frag.rows))
}

/// The schema a precursor-library table on disk records: version 2 when it carries
/// `predicted_im`, else version 1 (`artifact::fragment_library_precursors`). Read from the
/// parquet footer, so every site that records a library, whether predict-frag, an importer,
/// a DeepLC rewrite or a sub-library wrote it, records the version that was written.
pub fn precursors_schema(path: &str) -> Result<(&'static str, u32)> {
    let has_im = mumdia_io::table::column_names(path)?
        .iter()
        .any(|c| c == "predicted_im");
    Ok(artifact::fragment_library_precursors(has_im))
}

/// [`run`], returning each output's row count and the content hash its report records
/// (precursors, then fragments), so an orchestrator can record both artifacts without
/// reading and hashing them again.
pub fn run_hashed(p: PredictFragParams) -> Result<(Written, Written)> {
    let t0 = Instant::now();
    let t = TableFile::open(p.peptidoforms)?;
    let pf_id = t.u32("id")?;
    let base_id = t.u32("base_peptide_id")?;
    let pform = t.str("peptidoform")?;
    let charge = t.i32("charge")?;
    let label = t.str("label")?;
    let protein = t.str("protein")?;

    // Phase A: parse and compute fragments (no intensity yet). Rows are
    // independent, so map them in parallel; `collect` preserves row order, and
    // the sequential accumulation below reproduces the exact serial `raws` order
    // and `n_parse_err` count (raws is re-sorted by precursor m/z at Stage D
    // build, whose stable sort then breaks m/z ties by this identical order).
    #[allow(clippy::large_enum_variant)] // Raw dominates; boxing would churn the hot map/collect
    enum RowOut {
        Raw(Raw),
        ParseErr,
        Empty,
    }
    let outs: Vec<RowOut> = (0..t.nrows)
        .into_par_iter()
        .map(|row| {
            let parsed = match parse_peptidoform(&pform[row]) {
                Ok(p) => p,
                Err(_) => return RowOut::ParseErr,
            };
            let z = charge[row];
            let frags = if p.cfg.charge_by_basic_residues {
                // Composition cap: request every charge up to the smaller of the
                // precursor charge and the peptide's proton capacity, then keep
                // each fragment only at charges its own basic sites (Arg/His/Lys
                // within the fragment) plus its N-terminal amine can hold.
                let max_z = (1 + parsed.basic_residue_count() as i32).min(z).max(1);
                let all_charges: Vec<i32> = (1..=max_z).collect();
                let mut fr = parsed.fragments(&all_charges);
                fr.retain(|f| {
                    f.charge <= 1 + parsed.fragment_basic_sites(f.ion_type, f.ordinal) as i32
                });
                fr
            } else {
                let mut frag_charges = vec![1];
                if z >= p.cfg.charge2_from_precursor_charge {
                    frag_charges.push(2);
                }
                parsed.fragments(&frag_charges)
            };
            if frags.is_empty() {
                return RowOut::Empty;
            }
            RowOut::Raw(Raw {
                peptidoform_id: pf_id[row],
                base_peptide_id: base_id[row],
                peptidoform: pform[row].clone(),
                charge: z,
                label: label[row].clone(),
                protein: protein[row].clone(),
                precursor_mz: parsed.precursor_mz(z),
                frags,
                irt: 0.0,
                im: None,
                frag_int: Vec::new(),
                parsed,
            })
        })
        .collect();
    let mut raws: Vec<Raw> = Vec::with_capacity(t.nrows);
    let mut n_parse_err = 0u64;
    for o in outs {
        match o {
            RowOut::Raw(r) => raws.push(r),
            RowOut::ParseErr => n_parse_err += 1,
            RowOut::Empty => {}
        }
    }
    info!(
        candidates = raws.len(),
        parse_errors = n_parse_err,
        "predict-frag: parsed"
    );

    let Predictions {
        rt_id: rt_model_id,
        rt_missing,
        frag_id: frag_model_id,
        frag_missing,
        im,
    } = assign_predictions(&p, &mut raws)?;
    // The library carries 1/K0, and is the version-2 table, exactly when an ion-mobility
    // predictor ran; otherwise every byte is the version-1 library's.
    let has_im = im.is_some();
    let (im_model_id, im_missing) = im.unwrap_or_default();
    let model_identity = if has_im {
        format!("{rt_model_id}; {frag_model_id}; {im_model_id}")
    } else {
        format!("{rt_model_id}; {frag_model_id}")
    };

    // Coverage. A candidate a predictor returned nothing for is dropped, together with
    // every candidate sharing its pair key (base peptide, charge, modification set), so
    // a target and its paired decoy leave the library together and the exchangeability
    // the FDR rests on is kept. Substituting a value was the previous behaviour, iRT 0.0
    // for DeepLC and the native heuristic for MS2PIP, and it produced a plausible, finite
    // library whose rows came from an unrecorded mixture of predictors (docs/29 #17).
    let n_before_drop = raws.len() as u64;
    let missing = [
        rt_missing.as_slice(),
        frag_missing.as_slice(),
        im_missing.as_slice(),
    ];
    let (n_dropped_rows, n_dropped_pairs) = drop_unpredicted(&mut raws, &missing);
    // The rows the predictors actually missed, and the rows that went with them for
    // sharing a pair key. Reported separately because the second number is the cost of
    // the position-free key (see `rows_to_drop`) and the first is the sidecar's own miss
    // rate; one warning conflating them hid a worker returning nothing for a large batch.
    let n_direct = unpredicted_rows(&missing).len() as u64;
    let n_collateral = n_dropped_rows.saturating_sub(n_direct);
    if n_dropped_rows > 0 {
        tracing::warn!(
            candidates_dropped = n_dropped_rows,
            unpredicted = n_direct,
            dropped_with_their_pair = n_collateral,
            pairs_dropped = n_dropped_pairs,
            without_irt = rt_missing.len(),
            without_intensities = frag_missing.len(),
            without_im = im_missing.len(),
            "predict-frag: candidates without a prediction were dropped with their pairs \
             instead of receiving a substitute value; the counts are in the library report"
        );
    }
    if let Some(frac) = dropped_fraction_exceeded(n_dropped_rows, n_before_drop) {
        bail!(
            "predict-frag: {n_dropped_rows} of {n_before_drop} candidates ({:.2}%) have no \
             prediction ({n_direct} the predictor missed, {n_collateral} dropped with their \
             pair) and would silently leave the library. That is a failing sidecar rather \
             than a few unsupported peptidoforms; the ceiling is {:.0}%. Check the worker's \
             output above, then rerun.",
            frac * 100.0,
            MAX_DROPPED_FRACTION * 100.0
        );
    }
    if raws.is_empty() && (n_dropped_rows > 0 || n_parse_err > 0) {
        bail!(
            "predict-frag: no candidate received a prediction ({n_dropped_rows} dropped for \
             missing predictions); the predictor sidecar produced nothing usable"
        );
    }

    // Finite guard at the prediction sidecar boundary. A NaN/Inf predicted iRT or
    // fragment intensity from a misbehaving MS2PIP/DeepLC run would silently
    // corrupt the library and every downstream spectral-similarity feature (and
    // would misorder the top-N intensity sort below). Fail loudly instead. The
    // rescore sidecar boundary already guards its feature matrix the same way.
    for r in &raws {
        if !r.irt.is_finite() {
            bail!(
                "predict-frag: non-finite predicted iRT for '{}'",
                r.peptidoform
            );
        }
        if r.frag_int.iter().any(|v| !v.is_finite()) {
            bail!(
                "predict-frag: non-finite predicted fragment intensity for '{}'",
                r.peptidoform
            );
        }
    }

    // Keep top-N fragments by predicted intensity. Each candidate is independent
    // (operates only on its own frags/frag_int), so this parallelizes with no
    // cross-item state; the result per candidate is identical to the serial loop.
    raws.par_iter_mut().for_each(|r| {
        let mut order: Vec<usize> = (0..r.frags.len()).collect();
        order.sort_by(|&a, &b| r.frag_int[b].total_cmp(&r.frag_int[a]));
        order.truncate(p.cfg.top_n_fragments);
        order.sort_unstable();
        // In-place gather: `order` is ascending and `order[dst] >= dst`, so a
        // forward swap pass compacts the kept fragments into their original
        // relative order without cloning Fragments or reallocating the Vecs.
        for (dst, &src) in order.iter().enumerate() {
            r.frags.swap(dst, src);
            r.frag_int.swap(dst, src);
        }
        r.frags.truncate(order.len());
        r.frag_int.truncate(order.len());
    });
    raws.retain(|r| !r.frags.is_empty());

    // candidate_id = precursor-m/z sort key (docs/09_extract.md).
    // Parallel STABLE sort: rayon's par_sort_by is stable, so ties keep input order exactly
    // as the serial sort_by did. The fragment index requires this ascending-m/z ordering.
    raws.par_sort_by(|a, b| a.precursor_mz.total_cmp(&b.precursor_mz));

    let n = raws.len();
    let total_frags: usize = raws.iter().map(|r| r.frags.len()).sum();
    let (mut cid, mut pfid, mut baseid) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut pform_c, mut z_c, mut mz_c, mut irt_c) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut label_c, mut prot_c, mut nfrag_c) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut im_c: Vec<Option<f64>> = Vec::with_capacity(if has_im { n } else { 0 });
    let (mut f_cid, mut f_mz, mut f_int) = (
        Vec::with_capacity(total_frags),
        Vec::with_capacity(total_frags),
        Vec::with_capacity(total_frags),
    );
    let (mut f_name, mut f_type, mut f_ord, mut f_chg) = (
        Vec::with_capacity(total_frags),
        Vec::with_capacity(total_frags),
        Vec::with_capacity(total_frags),
        Vec::with_capacity(total_frags),
    );

    for (candidate_id, r) in raws.into_iter().enumerate() {
        let candidate_id = candidate_id as u32;
        cid.push(candidate_id);
        pfid.push(r.peptidoform_id);
        baseid.push(r.base_peptide_id);
        pform_c.push(r.peptidoform);
        z_c.push(r.charge);
        mz_c.push(r.precursor_mz);
        irt_c.push(r.irt);
        label_c.push(r.label);
        prot_c.push(r.protein);
        nfrag_c.push(r.frags.len() as i32);
        if has_im {
            im_c.push(r.im);
        }
        for (fr, fi) in r.frags.into_iter().zip(r.frag_int) {
            f_cid.push(candidate_id);
            f_mz.push(fr.mz);
            f_int.push(fi);
            f_name.push(fr.name());
            f_type.push(fr.ion_type.symbol().to_string());
            f_ord.push(fr.ordinal as i32);
            f_chg.push(fr.charge);
        }
    }

    let mut prec_cols = vec![
        Col::U32("candidate_id".into(), cid),
        Col::U32("peptidoform_id".into(), pfid),
        Col::U32("base_peptide_id".into(), baseid),
        Col::Str("peptidoform".into(), pform_c),
        Col::I32("charge".into(), z_c),
        Col::F64("precursor_mz".into(), mz_c),
        Col::F32("predicted_irt".into(), irt_c),
        Col::Str("label".into(), label_c),
        Col::Str("protein".into(), prot_c),
        Col::I32("n_fragments".into(), nfrag_c),
    ];
    if has_im {
        // Version 2 (`artifact::fragment_library_precursors`). Nullable Float64, matching an
        // imported library's column, although a predicted library has no null.
        prec_cols.push(Col::OptF64("predicted_im".into(), im_c));
    }
    let prec_written = write_table_hashed(p.out_precursors, prec_cols)?;
    // Fragment cardinality: distinct precursors sharing each fragment m/z (0.01 Da
    // bin), matching the imported-library path. Computed once over the whole
    // library so downstream interference-aware feature/quant selection reads a
    // deterministic precomputed column instead of a runtime heuristic. Diagnostic;
    // no consumer yet.
    let f_card = fragment_cardinality(&f_cid, &f_mz);
    let frag_written = write_table_hashed(
        p.out_fragments,
        vec![
            Col::U32("candidate_id".into(), f_cid),
            Col::F64("mz".into(), f_mz),
            Col::F32("predicted_intensity".into(), f_int),
            Col::Str("name".into(), f_name),
            Col::Str("ion_type".into(), f_type),
            Col::I32("ordinal".into(), f_ord),
            Col::I32("frag_charge".into(), f_chg),
            Col::I32("cardinality".into(), f_card),
        ],
    )?;

    let (n_prec, n_frag) = (prec_written.rows, frag_written.rows);
    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("candidates".to_string(), json!(n_prec));
    stats.insert("fragments".to_string(), json!(n_frag));
    stats.insert("parse_errors".to_string(), json!(n_parse_err));
    stats.insert(
        "candidates_dropped_with_their_pair".to_string(),
        json!(n_collateral),
    );
    stats.insert(
        "candidates_dropped_unpredicted".to_string(),
        json!(n_dropped_rows),
    );
    stats.insert(
        "pairs_dropped_unpredicted".to_string(),
        json!(n_dropped_pairs),
    );
    let mut params = json!({"top_n": p.cfg.top_n_fragments, "ms2pip_model": p.cfg.ms2pip_model,
                            "rt_predictor": format!("{:?}", p.cfg.rt_predictor),
                            "fragment_predictor": format!("{:?}", p.cfg.predictor)});
    if has_im {
        // Recorded only when 1/K0 was predicted, so a library without it keeps its report.
        stats.insert("without_im".to_string(), json!(im_missing.len() as u64));
        params["im_predictor"] = json!(format!("{:?}", ImPredictorKind::Im2deep));
    }
    let mut written: Vec<Written> = Vec::with_capacity(2);
    for (path, schema, file) in [
        (
            p.out_precursors,
            artifact::fragment_library_precursors(has_im),
            prec_written,
        ),
        (
            p.out_fragments,
            artifact::FRAGMENT_LIBRARY_FRAGMENTS,
            frag_written,
        ),
    ] {
        let report = ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "predict-frag".to_string(),
            rows: file.rows,
            // Both tables were hashed while they were written.
            content_hash: file.content_hash,
            params: params.clone(),
            stats: stats.clone(),
            model_identity: Some(model_identity.clone()),
            elapsed_ms: elapsed,
        };
        report.write_for(path)?;
        written.push(report.written());
    }

    info!(
        candidates = n_prec,
        fragments = n_frag,
        elapsed_ms = elapsed,
        "predict-frag: done"
    );
    let frag = written.pop().expect("two reports written");
    let prec = written.pop().expect("two reports written");
    Ok((prec, frag))
}

/// What the predictors returned: each model's identity and the candidate indices it
/// returned nothing for (the caller drops those with their pairs). `im` is `None` when no
/// ion-mobility predictor ran.
struct Predictions {
    rt_id: String,
    rt_missing: Vec<usize>,
    frag_id: String,
    frag_missing: Vec<usize>,
    im: Option<(String, Vec<usize>)>,
}

/// Predicted iRT, fragment intensities and, with `predict_frag.im_predictor`, 1/K0 for
/// every candidate.
///
/// The sidecar workers run one after another by default: DeepLC, then the fragment
/// intensities, then IM2Deep. When `MUMDIA_PREDICT_FRAG_CONCURRENT=1` is set and at least
/// two of them run, they run at the same time: they write disjoint fields of each
/// candidate, and each gets exactly the request and thread count it gets when run alone,
/// so the library is byte-identical either way (measured on the fixture library with
/// DeepLC 4.5.0 and MS2PIP 4.2.0: 9.1 s against 14.7 s; IM2Deep alongside them is not
/// measured).
///
/// Off by default. Each worker sizes itself from the engine's whole thread count (MS2PIP
/// starts that many processes, DeepLC and IM2Deep take that many torch threads), so at once
/// they ask for several times the CPUs and hold every worker's memory; the saving on the
/// fixture is the workers' start-up, and at scale, where they are CPU-bound (DeepLC 19 min
/// and MS2PIP 35-39 min on the 9.8M-peptidoform HYE FASTA library, one after the other),
/// neither the wall time nor the process-tree peak has been measured. Splitting the thread
/// budget instead would change DeepLC's torch thread count and with it the last bits of
/// its predictions. Validate on a large FASTA build (wall time, peak, and a byte
/// comparison of both library tables) before turning it on.
fn assign_predictions(p: &PredictFragParams, raws: &mut [Raw]) -> Result<Predictions> {
    let rt_req = RtRequest::new(p, raws)?;
    let frag_req = FragRequest::new(p, raws)?;
    let im_req = ImRequest::new(p, raws)?;
    let concurrent = std::env::var("MUMDIA_PREDICT_FRAG_CONCURRENT")
        .is_ok_and(|v| !v.trim().is_empty() && v.trim() != "0");
    let deeplc = match &rt_req {
        RtRequest::Deeplc(rt) => Some(rt),
        _ => None,
    };
    let n_workers = [deeplc.is_some(), frag_req.is_some(), im_req.is_some()]
        .iter()
        .filter(|&&w| w)
        .count();
    let (rt_out, frag_out, im_out) = if concurrent && n_workers >= 2 {
        if im_req.is_none() {
            info!(
                "predict-frag: running the DeepLC and the fragment-intensity workers at the \
                 same time (MUMDIA_PREDICT_FRAG_CONCURRENT)"
            );
        } else {
            info!(
                workers = n_workers,
                "predict-frag: running the prediction workers at the same time \
                 (MUMDIA_PREDICT_FRAG_CONCURRENT)"
            );
        }
        fn joined<T>(h: std::thread::ScopedJoinHandle<'_, Result<T>>, who: &str) -> Result<T> {
            h.join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("the {who} worker thread panicked")))
        }
        let (rt, frag, im) = std::thread::scope(|scope| {
            let rt = deeplc.map(|rt| scope.spawn(|| rt.call(p)));
            let im = im_req.as_ref().map(|r| scope.spawn(|| r.call(p)));
            let frag = frag_req.as_ref().map(|fr| fr.call(p));
            if let Some(Err(e)) = &frag {
                // Said now rather than after the other workers are joined, which on a large
                // library is many minutes later: the stage fails either way, and the other
                // workers cannot be interrupted from here.
                tracing::error!(
                    error = %format!("{e:#}"),
                    "predict-frag: the fragment-intensity worker failed; waiting for the \
                     other workers to exit before the stage fails"
                );
            }
            let rt = rt.map(|h| joined(h, "DeepLC"));
            let im = im.map(|h| joined(h, "IM2Deep"));
            (rt, frag, im)
        });
        // DeepLC's error first, as when the workers run one after the other.
        (rt.transpose()?, frag.transpose()?, im.transpose()?)
    } else {
        // DeepLC first, as before: a failing DeepLC stops the stage before MS2PIP runs, and
        // a failing fragment worker before IM2Deep runs.
        let rt = deeplc.map(|rt| rt.call(p)).transpose()?;
        let frag = frag_req.as_ref().map(|fr| fr.call(p)).transpose()?;
        let im = im_req.as_ref().map(|r| r.call(p)).transpose()?;
        (rt, frag, im)
    };
    let (rt_id, rt_missing) = rt_req.apply(raws, rt_out)?;
    let (frag_id, frag_missing) = match (frag_req, frag_out) {
        (Some(fr), Some(map)) => fr.apply(p, raws, map)?,
        _ => {
            let m = NativeFrag;
            // Per-row and independent: each candidate's intensities depend only on its own
            // parsed peptidoform and fragment list, with no cross-row reduction, so this is
            // bit-identical in parallel (no float accumulation order to change).
            raws.par_iter_mut().for_each(|r| {
                r.frag_int = m.predict_intensities(&r.parsed, &r.frags);
            });
            (m.identity(), Vec::new())
        }
    };
    let im = match (im_req, im_out) {
        (Some(req), Some(out)) => Some(req.apply(raws, out)),
        _ => None,
    };
    Ok(Predictions {
        rt_id,
        rt_missing,
        frag_id,
        frag_missing,
        im,
    })
}

/// The IM2Deep request: one prediction per unique (peptidoform, charge). 1/K0 depends on
/// the charge, unlike the iRT, and the precursor m/z the worker converts the CCS at follows
/// from the pair.
struct ImRequest {
    python: String,
    script: String,
    keys: ImKeys,
}

/// The unique (peptidoform, charge) pairs of the candidates, in first-seen row order, and
/// each row's index into them. The request id of a pair is its index, so the request does
/// not depend on hashing.
#[derive(Debug, Default, PartialEq)]
struct ImKeys {
    peps: Vec<String>,
    charges: Vec<i32>,
    mzs: Vec<f64>,
    row_ids: Vec<u32>,
}

impl ImKeys {
    fn of<'a>(rows: impl IntoIterator<Item = (&'a str, i32, f64)>) -> Self {
        let mut uniq: HashMap<(&'a str, i32), u32> = HashMap::new();
        let mut keys = Self::default();
        for (peptidoform, charge, mz) in rows {
            let next = keys.peps.len() as u32;
            let id = *uniq.entry((peptidoform, charge)).or_insert_with(|| {
                keys.peps.push(peptidoform.to_string());
                keys.charges.push(charge);
                keys.mzs.push(mz);
                next
            });
            keys.row_ids.push(id);
        }
        keys
    }

    fn ids(&self) -> Vec<u32> {
        (0..self.peps.len() as u32).collect()
    }
}

impl ImRequest {
    /// `None` when the build predicts no 1/K0 (`predict_frag.im_predictor` resolves to
    /// `none`).
    fn new(p: &PredictFragParams, raws: &[Raw]) -> Result<Option<Self>> {
        match p.cfg.im_predictor_resolved() {
            ImPredictorKind::None => {
                if p.cfg.im_predictor == ImPredictorKind::Auto {
                    info!(
                        "predict-frag: predict_frag.im_predictor = auto and no IM2Deep \
                         interpreter resolved, so the library carries no predicted 1/K0 (set \
                         predict_frag.im2deep_python or MUMDIA_PYTHON_IM2DEEP)"
                    );
                }
                Ok(None)
            }
            // `im_predictor_resolved` never returns `Auto`.
            ImPredictorKind::Im2deep | ImPredictorKind::Auto => {
                let python = p.cfg.im2deep_python.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("im_predictor=im2deep requires predict_frag.im2deep_python")
                })?;
                Ok(Some(Self {
                    python: python.to_string(),
                    script: crate::sidecar::resolve_script(
                        &p.cfg.sidecar_script_dir,
                        "im2deep_worker.py",
                    ),
                    keys: ImKeys::of(
                        raws.iter()
                            .map(|r| (r.peptidoform.as_str(), r.charge, r.precursor_mz)),
                    ),
                }))
            }
        }
    }

    fn call(&self, p: &PredictFragParams) -> Result<HashMap<u32, f64>> {
        sidecar::run_im2deep(
            &self.python,
            &self.script,
            p.work_dir,
            &self.keys.ids(),
            &self.keys.peps,
            &self.keys.charges,
            &self.keys.mzs,
        )
    }

    /// Assign every candidate its 1/K0. Returns the model id and the candidates the worker
    /// returned nothing for, which the caller drops with their pairs.
    fn apply(self, raws: &mut [Raw], out: HashMap<u32, f64>) -> (String, Vec<usize>) {
        let missing = assign_im(raws.iter_mut().map(|r| &mut r.im), &self.keys, &out);
        // The single-conformer model, uncalibrated, like the DeepLC base model's iRT.
        let id = sidecar::model_identity(Some(&self.python), "im2deep", "uni");
        (id, missing)
    }
}

/// Write each row's 1/K0 from the worker's `id -> value` map into `slots` (one per row, in
/// the order the keys were built from). Returns the rows without a value.
fn assign_im<'a>(
    slots: impl Iterator<Item = &'a mut Option<f64>>,
    keys: &ImKeys,
    out: &HashMap<u32, f64>,
) -> Vec<usize> {
    let mut missing = Vec::new();
    for (i, (slot, id)) in slots.zip(&keys.row_ids).enumerate() {
        match out.get(id) {
            Some(&v) => *slot = Some(v),
            None => missing.push(i),
        }
    }
    missing
}

/// How every candidate gets its iRT.
enum RtRequest {
    Native,
    /// The native model's iRT as a stand-in for DeepLC's (`PredictFragParams::rt_placeholder`).
    Placeholder,
    Deeplc(DeeplcRequest),
}

/// The DeepLC request: one prediction per unique peptidoform (RT is charge-independent).
struct DeeplcRequest {
    python: String,
    script: String,
    uniq: HashMap<String, u32>,
    ids: Vec<u32>,
    peps: Vec<String>,
}

impl RtRequest {
    fn new(p: &PredictFragParams, raws: &[Raw]) -> Result<Self> {
        match p.cfg.rt_predictor {
            RtPredictorKind::Native => Ok(Self::Native),
            RtPredictorKind::Deeplc if p.rt_placeholder => {
                info!(
                    "predict-frag: skipping the DeepLC pass: the multi-head calibration \
                     re-predicts every row of this library before anything reads its iRT \
                     (predict_frag.defer_deeplc_to_multihead). The library carries the native \
                     model's iRT as a placeholder until then, and the run fails if the \
                     calibration leaves any row un-predicted"
                );
                Ok(Self::Placeholder)
            }
            RtPredictorKind::Deeplc => {
                let python = p.cfg.deeplc_python.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("rt_predictor=deeplc requires predict_frag.deeplc_python")
                })?;
                let script =
                    crate::sidecar::resolve_script(&p.cfg.sidecar_script_dir, "deeplc_worker.py");
                // Dedup by peptidoform (RT is charge-independent).
                let mut uniq: HashMap<String, u32> = HashMap::new();
                let (mut ids, mut peps) = (Vec::new(), Vec::new());
                for r in raws.iter() {
                    if !uniq.contains_key(&r.peptidoform) {
                        let id = uniq.len() as u32;
                        uniq.insert(r.peptidoform.clone(), id);
                        ids.push(id);
                        peps.push(r.peptidoform.clone());
                    }
                }
                Ok(Self::Deeplc(DeeplcRequest {
                    python: python.to_string(),
                    script,
                    uniq,
                    ids,
                    peps,
                }))
            }
        }
    }

    /// Assign the iRT. Returns the model id and the candidates without a prediction.
    fn apply(
        self,
        raws: &mut [Raw],
        out: Option<HashMap<u32, f32>>,
    ) -> Result<(String, Vec<usize>)> {
        match self {
            Self::Native | Self::Placeholder => {
                let m = NativeRt;
                // Per-row and independent (no cross-row reduction) -> bit-identical in parallel.
                raws.par_iter_mut().for_each(|r| {
                    r.irt = m.predict_irt(&r.parsed);
                });
                let id = if matches!(self, Self::Placeholder) {
                    format!(
                        "{} (placeholder: DeepLC deferred to the multi-head calibration)",
                        m.identity()
                    )
                } else {
                    m.identity()
                };
                Ok((id, Vec::new()))
            }
            Self::Deeplc(req) => {
                let out = out.expect("a DeepLC request is called before it is applied");
                let mut missing = Vec::new();
                for (i, r) in raws.iter_mut().enumerate() {
                    let uid = req.uniq[&r.peptidoform];
                    match out.get(&uid) {
                        Some(&v) => r.irt = v,
                        None => missing.push(i),
                    }
                }
                // The installed DeepLC version, not a family label: two libraries predicted by
                // different DeepLC releases are different libraries (docs/30, model identity).
                let version = sidecar::require_deeplc_version(&req.python)?;
                Ok((format!("deeplc-{version}-base"), missing))
            }
        }
    }
}

impl DeeplcRequest {
    fn call(&self, p: &PredictFragParams) -> Result<HashMap<u32, f32>> {
        sidecar::run_deeplc(
            &self.python,
            &self.script,
            p.work_dir,
            &self.ids,
            &self.peps,
        )
    }
}

/// Which fragment-intensity worker a [`FragRequest`] calls.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FragModel {
    Ms2pip,
    Peptdeep,
}

/// A fragment-intensity sidecar request: one row per candidate.
struct FragRequest {
    model: FragModel,
    python: String,
    script: String,
    ids: Vec<u32>,
    peps: Vec<String>,
    charges: Vec<i32>,
}

impl FragRequest {
    /// `None` for the native predictor, which needs no sidecar.
    fn new(p: &PredictFragParams, raws: &[Raw]) -> Result<Option<Self>> {
        let (model, python, worker) = match p.cfg.predictor {
            FragPredictorKind::Native => return Ok(None),
            FragPredictorKind::Ms2pip => (
                FragModel::Ms2pip,
                p.cfg.ms2pip_python.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("predictor=ms2pip requires predict_frag.ms2pip_python")
                })?,
                "ms2pip_worker.py",
            ),
            FragPredictorKind::Peptdeep => (
                FragModel::Peptdeep,
                p.cfg.peptdeep_python.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("predictor=peptdeep requires predict_frag.peptdeep_python")
                })?,
                "peptdeep_worker.py",
            ),
        };
        Ok(Some(Self {
            model,
            python: python.to_string(),
            script: crate::sidecar::resolve_script(&p.cfg.sidecar_script_dir, worker),
            ids: (0..raws.len() as u32).collect(),
            peps: raws.iter().map(|r| r.peptidoform.clone()).collect(),
            charges: raws.iter().map(|r| r.charge).collect(),
        }))
    }

    fn call(&self, p: &PredictFragParams) -> Result<sidecar::FragmentIntensityMap> {
        match self.model {
            FragModel::Ms2pip => sidecar::run_ms2pip(
                &self.python,
                &self.script,
                p.work_dir,
                &self.ids,
                &self.peps,
                &self.charges,
                &p.cfg.ms2pip_model,
            ),
            FragModel::Peptdeep => sidecar::run_peptdeep(
                &self.python,
                &self.script,
                p.work_dir,
                &self.ids,
                &self.peps,
                &self.charges,
                &p.cfg.peptdeep_model,
                p.cfg.peptdeep_nce,
                &p.cfg.peptdeep_instrument,
            ),
        }
    }

    /// Assign every fragment its intensity from the worker's predictions. Returns the model
    /// id and the candidates the worker returned nothing for.
    fn apply(
        self,
        p: &PredictFragParams,
        raws: &mut [Raw],
        map: sidecar::FragmentIntensityMap,
    ) -> Result<(String, Vec<usize>)> {
        if map.is_empty() {
            match self.model {
                FragModel::Ms2pip => bail!("MS2PIP returned no predictions"),
                FragModel::Peptdeep => bail!("AlphaPeptDeep returned no predictions"),
            }
        }
        let native = NativeFrag;
        // Parallel across rows: each row writes only its own `frag_int` and reads only its
        // own parsed peptidoform, fragment list and worker entry. `par_iter_mut` preserves the
        // element-to-index mapping, and the per-row float work (the native charge-2 fallback
        // and the per-charge-group max-normalizations) is self-contained, so this is
        // bit-identical to the serial loop. The worker call already happened -- what is
        // parallelized here is the per-row native prediction and normalization, which is real
        // CPU work, not sidecar wait.
        //
        // A candidate the worker returned nothing for is reported to the caller, which drops
        // it with its pair; it used to receive the native heuristic silently, under a
        // library-wide model identity (docs/29 #17). AlphaPeptDeep goes through the same
        // `ms2pip_values`: its worker always emits both fragment charges, so
        // `model_has_charge2` holds and every fragment is on one scale.
        let covered: Vec<bool> = raws
            .par_iter_mut()
            .enumerate()
            .map(|(i, r)| match map.get(&(i as u32)) {
                Some(per) if !per.is_empty() => {
                    let keys: Vec<(u8, u16, u8)> = r
                        .frags
                        .iter()
                        .map(|fr| {
                            (
                                fr.ion_type.symbol() as u8,
                                fr.ordinal as u16,
                                fr.charge.clamp(1, 255) as u8,
                            )
                        })
                        .collect();
                    let nat = native.predict_intensities(&r.parsed, &r.frags);
                    r.frag_int = ms2pip_values(&keys, per, &nat);
                    true
                }
                _ => {
                    r.frag_int = vec![0.0; r.frags.len()];
                    false
                }
            })
            .collect();
        let missing: Vec<usize> = covered
            .iter()
            .enumerate()
            .filter(|(_, &c)| !c)
            .map(|(i, _)| i)
            .collect();
        let id = match self.model {
            FragModel::Ms2pip => {
                let version = sidecar::module_version(&self.python, "ms2pip")
                    .unwrap_or_else(|| "unknown".into());
                format!("ms2pip-{version}-{}", p.cfg.ms2pip_model)
            }
            FragModel::Peptdeep => {
                let version = sidecar::module_version(&self.python, "peptdeep")
                    .unwrap_or_else(|| "unknown".into());
                // The identity carries what changes the numbers: the model, the collision
                // energy and the instrument. Two libraries built at different NCE are not the
                // same library, and the manifest has to be able to say so.
                format!(
                    "peptdeep-{version}-{}-nce{}-{}",
                    p.cfg.peptdeep_model, p.cfg.peptdeep_nce, p.cfg.peptdeep_instrument
                )
            }
        };
        Ok((id, missing))
    }
}

/// The modification content of a ProForma peptidoform, order-free: every bracketed
/// token, sorted and joined. Two peptidoforms with the same base peptide, charge and
/// signature are the same precursor on the target and decoy side.
fn mod_signature(peptidoform: &str) -> String {
    let mut mods: Vec<&str> = Vec::new();
    let mut rest = peptidoform;
    while let Some(open) = rest.find('[') {
        match rest[open..].find(']') {
            Some(close) => {
                mods.push(&rest[open + 1..open + close]);
                rest = &rest[open + close + 1..];
            }
            None => break,
        }
    }
    mods.sort_unstable();
    mods.join(",")
}

/// The key under which a target and its paired decoy are one precursor.
fn pair_key(base_peptide_id: u32, charge: i32, peptidoform: &str) -> (u32, i32, String) {
    (base_peptide_id, charge, mod_signature(peptidoform))
}

/// The largest fraction of the library that may be dropped for want of a prediction
/// before the build is treated as a failed sidecar rather than a few unsupported
/// peptidoforms.
///
/// The only previous guard fired when the library was emptied entirely, so a worker that
/// returned nothing for 5% of a 9.8M-peptidoform batch removed those rows, plus every
/// positional isomer sharing their signature, behind one warning (docs/31 F10).
const MAX_DROPPED_FRACTION: f64 = 0.02;

/// The dropped fraction when it exceeds [`MAX_DROPPED_FRACTION`], else `None`.
fn dropped_fraction_exceeded(dropped: u64, total: u64) -> Option<f64> {
    if total == 0 || dropped == 0 {
        return None;
    }
    let frac = dropped as f64 / total as f64;
    (frac > MAX_DROPPED_FRACTION).then_some(frac)
}

/// Which rows to drop so that every row sharing a pair key with an unpredicted row goes
/// with it. `keys[i]` is the pair key of row `i`; `missing` lists unpredicted rows.
///
/// The key is deliberately position-free, and that over-deletes: `M[Oxidation]PEPTIDEMK`
/// and `MPEPTIDEM[Oxidation]K` share a base peptide, a charge and a modification
/// multiset, so a miss on either removes both. Making the key position-aware would be
/// worse, not better: a reverse decoy carries its modifications at mirrored positions, so
/// a positional key would stop matching a target to its decoy and a target could be
/// dropped while its decoy stayed, which is an FDR defect rather than a sensitivity one.
/// The collateral is counted separately and bounded by [`MAX_DROPPED_FRACTION`] instead.
fn rows_to_drop(keys: &[(u32, i32, String)], missing: &[usize]) -> Vec<bool> {
    let doomed: std::collections::HashSet<&(u32, i32, String)> =
        missing.iter().map(|&i| &keys[i]).collect();
    keys.iter().map(|k| doomed.contains(k)).collect()
}

/// The rows any predictor missed, ascending and each once. `missing` holds one list per
/// predictor (iRT, intensities, 1/K0).
fn unpredicted_rows(missing: &[&[usize]]) -> Vec<usize> {
    let mut rows: Vec<usize> = missing.iter().flat_map(|m| m.iter().copied()).collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Drop every candidate without a prediction together with its pair. `missing` holds
/// one list of unpredicted rows per predictor. Returns the number of rows and of
/// distinct pair keys removed.
fn drop_unpredicted(raws: &mut Vec<Raw>, missing: &[&[usize]]) -> (u64, u64) {
    if missing.iter().all(|m| m.is_empty()) {
        return (0, 0);
    }
    let keys: Vec<(u32, i32, String)> = raws
        .iter()
        .map(|r| pair_key(r.base_peptide_id, r.charge, &r.peptidoform))
        .collect();
    let missing = unpredicted_rows(missing);
    let drop = rows_to_drop(&keys, &missing);
    let n_pairs = missing
        .iter()
        .map(|&i| &keys[i])
        .collect::<std::collections::HashSet<_>>()
        .len() as u64;
    let before = raws.len();
    let mut i = 0usize;
    raws.retain(|_| {
        let keep = !drop[i];
        i += 1;
        keep
    });
    ((before - raws.len()) as u64, n_pairs)
}

/// One candidate's fragment intensities from its MS2PIP predictions.
///
/// `keys` are the candidate's fragments as `(ion byte, ordinal, charge)`, `per` the
/// worker's predictions for it, `native` the heuristic intensities for the same
/// fragments. Two regimes, decided by what the model emitted:
///
/// - **Charge-1 series only** (HCD2021 and the other single-charge models): charge-1
///   fragments take the prediction (0.0 when the ordinal is missing), charge-2 fragments
///   take the native heuristic, and each charge group is max-normalised to its own peak,
///   because MS2PIP's TIC fractions (~0.02-0.3) and the heuristic (~0.19-0.5) live on
///   different scales and ranking them together buries the predictions. This is the
///   historical behaviour, unchanged.
/// - **Charge-2 series present** (`HCDch2`, `CIDch2`): every fragment takes the
///   prediction for its own charge, a missing entry is 0.0 for either charge, and the
///   whole candidate is normalised to one peak, since all values share the model's scale.
///   Measured on the HYE FASTA library with the first regime, 78.6% of the top-6
///   fragments were charge-2 heuristics (each group's peak normalised to 1.0, so the
///   heuristics crowded out the predictions) and the seed search separated targets from
///   decoys no better than chance (41.6% decoys in the top 1,000 by score against 0% with
///   the DIA-NN library on the same spectra).
fn ms2pip_values(
    keys: &[(u8, u16, u8)],
    per: &HashMap<(u8, u16, u8), f32>,
    native: &[f32],
) -> Vec<f32> {
    let model_has_charge2 = per.keys().any(|k| k.2 >= 2);
    let mut vals: Vec<f32> = keys
        .iter()
        .enumerate()
        .map(|(k, key)| {
            if key.2 == 1 || model_has_charge2 {
                *per.get(key).unwrap_or(&0.0)
            } else {
                native[k]
            }
        })
        .collect();
    if model_has_charge2 {
        let m = vals.iter().copied().fold(0.0f32, f32::max);
        if m > 0.0 {
            for v in &mut vals {
                *v /= m;
            }
        }
    } else {
        let gmax = |want2: bool| {
            keys.iter()
                .zip(&vals)
                .filter(|(key, _)| (key.2 >= 2) == want2)
                .map(|(_, v)| *v)
                .fold(0.0f32, f32::max)
        };
        let (m1, m2) = (gmax(false), gmax(true));
        for (k, key) in keys.iter().enumerate() {
            let m = if key.2 >= 2 { m2 } else { m1 };
            if m > 0.0 {
                vals[k] /= m;
            }
        }
    }
    vals
}

/// Count distinct precursors (candidate_ids) whose fragments fall in each 0.01 Da
/// m/z bin, and return that count per fragment row aligned to `mz`. This is the
/// library fragment cardinality: a low value marks a clean, quantification-
/// friendly ion, a high value an interference-prone non-unique ion. Deterministic
/// (ordered maps, no HashMap iteration) and matched to the imported-library
/// binning in `scripts/import_diann_lib.py`.
fn fragment_cardinality(cid: &[u32], mz: &[f64]) -> Vec<i32> {
    // Sort-based count rather than a BTreeMap<bin, BTreeSet<candidate>>: at library scale
    // (~86M fragment rows) that map allocated a tree node per bin plus a whole set per bin
    // and did O(log n) descents per row twice over. Sorting (bin, candidate) pairs once and
    // counting distinct candidates in each run of equal bins is the same computation with
    // flat memory. Values are unchanged: the bin key and the distinct-precursor-per-bin
    // definition are identical, and the result stays deterministic (total order on the
    // sorted pairs, no hashing).
    let bin = |m: f64| (m * 100.0).round() as i64;
    let n = mz.len();
    let mut pairs: Vec<(i64, u32)> = (0..n).map(|i| (bin(mz[i]), cid[i])).collect();
    pairs.par_sort_unstable();
    // Walk each equal-bin run once, counting distinct candidate_ids (sorted, so distinct
    // == adjacent-different), and record the count against the bin.
    let mut counts: Vec<(i64, i32)> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let b = pairs[i].0;
        let mut j = i;
        let mut distinct = 0i32;
        let mut last: Option<u32> = None;
        while j < n && pairs[j].0 == b {
            if last != Some(pairs[j].1) {
                distinct += 1;
                last = Some(pairs[j].1);
            }
            j += 1;
        }
        counts.push((b, distinct));
        i = j;
    }
    // counts is ascending in `b`, so each row's bin resolves by binary search.
    mz.iter()
        .map(|m| {
            let b = bin(*m);
            let k = counts.partition_point(|&(bb, _)| bb < b);
            counts[k].1
        })
        .collect()
}

#[cfg(test)]
mod cardinality_tests {
    use super::{
        dropped_fraction_exceeded, fragment_cardinality, mod_signature, ms2pip_values, pair_key,
        rows_to_drop, MAX_DROPPED_FRACTION,
    };
    use std::collections::HashMap;

    #[test]
    fn unpredicted_candidates_are_dropped_with_their_pairs() {
        // Pair key = (base peptide, charge, modification set): the target PEPTIDEK/2 and
        // its reversed decoy share base 7 and charge 2 with no modifications; the oxidised
        // form is a different precursor and stays.
        assert_eq!(mod_signature("PEPTM[Oxidation]IDEK"), "Oxidation");
        assert_eq!(
            mod_signature("C[Carbamidomethyl]M[Oxidation]C[Carbamidomethyl]K"),
            "Carbamidomethyl,Carbamidomethyl,Oxidation"
        );
        assert_eq!(mod_signature("[Acetyl]-PEPTIDEK"), "Acetyl");
        assert_eq!(mod_signature("PEPTIDEK"), "");
        let keys = vec![
            pair_key(7, 2, "PEPTIDEK"),
            pair_key(7, 2, "KEDITPEP"),
            pair_key(7, 2, "PEPTM[Oxidation]IDEK"),
            pair_key(7, 3, "PEPTIDEK"),
            pair_key(8, 2, "SAMPLER"),
        ];
        // Only the target PEPTIDEK/2 lacked a prediction: it and its decoy go, the
        // oxidised form, the charge-3 form and the other peptide stay.
        assert_eq!(
            rows_to_drop(&keys, &[0]),
            vec![true, true, false, false, false]
        );
        assert_eq!(rows_to_drop(&keys, &[]), vec![false; 5]);
    }

    #[test]
    fn ms2pip_values_keep_the_two_group_regime_for_single_charge_models() {
        // b2(1), y3(1), y3(2), y4(2): the model emitted charge 1 only.
        let keys = [(b'b', 2, 1), (b'y', 3, 1), (b'y', 3, 2), (b'y', 4, 2)];
        let mut per = HashMap::new();
        per.insert((b'b', 2, 1), 0.05f32);
        per.insert((b'y', 3, 1), 0.20f32);
        let native = [0.9f32, 0.9, 0.5, 0.25];
        let v = ms2pip_values(&keys, &per, &native);
        // Charge-1 from the model, normalised to its own peak; charge-2 from the
        // heuristic, normalised to its own peak.
        assert_eq!(v, vec![0.25, 1.0, 1.0, 0.5]);
        // A charge-1 ordinal the model did not emit is 0.0, not the heuristic.
        let keys2 = [(b'b', 2, 1), (b'b', 5, 1)];
        let v2 = ms2pip_values(&keys2, &per, &[0.9, 0.9]);
        assert_eq!(v2, vec![1.0, 0.0]);
    }

    #[test]
    fn a_large_unpredicted_fraction_is_a_failure_not_a_warning() {
        // docs/31 F10: the only previous guard fired when the library was emptied, so a
        // worker returning nothing for a large batch removed those rows plus every
        // positional isomer sharing their signature behind one warning.
        assert_eq!(dropped_fraction_exceeded(0, 1000), None);
        assert_eq!(dropped_fraction_exceeded(0, 0), None);
        assert_eq!(
            dropped_fraction_exceeded(5, 1000),
            None,
            "a few misses are fine"
        );
        let frac = dropped_fraction_exceeded(500_000, 9_800_000).expect("5% must be refused");
        assert!(frac > MAX_DROPPED_FRACTION);
        // The boundary itself is accepted; only exceeding it is refused.
        assert_eq!(dropped_fraction_exceeded(20, 1000), None);
        assert!(dropped_fraction_exceeded(21, 1000).is_some());
    }

    #[test]
    fn a_positional_isomer_still_shares_its_pair_key_by_design() {
        // Deliberate over-deletion, documented on `rows_to_drop`: a positional key would
        // stop matching a reverse decoy to its target, which is an FDR defect. The test
        // pins the trade so a future change has to argue with it.
        let a = pair_key(7, 2, "M[Oxidation]PEPTIDEMK");
        let b = pair_key(7, 2, "MPEPTIDEM[Oxidation]K");
        assert_eq!(a, b);
        let keys = vec![a.clone(), b, pair_key(7, 3, "M[Oxidation]PEPTIDEMK")];
        let drop = rows_to_drop(&keys, &[0]);
        assert_eq!(
            drop,
            vec![true, true, false],
            "the other charge is a different precursor"
        );
    }

    #[test]
    fn ms2pip_values_use_one_scale_when_the_model_emitted_charge_2() {
        let keys = [(b'b', 2, 1), (b'y', 3, 1), (b'y', 3, 2), (b'y', 4, 2)];
        let mut per = HashMap::new();
        per.insert((b'b', 2, 1), 0.05f32);
        per.insert((b'y', 3, 1), 0.20f32);
        per.insert((b'y', 3, 2), 0.10f32);
        // y4(2) is absent from the predictions: 0.0, never the heuristic.
        let native = [0.9f32, 0.9, 0.9, 0.9];
        let v = ms2pip_values(&keys, &per, &native);
        assert_eq!(v, vec![0.25, 1.0, 0.5, 0.0]);
    }

    #[test]
    fn counts_distinct_precursors_per_mz_bin() {
        // Fragment m/z 100.00 shared by precursors 0 and 1 (cardinality 2);
        // 100.004 falls in the same 0.01 Da bin (still 2); 200.00 is unique to
        // precursor 2 (cardinality 1). A precursor counted once per bin.
        let cid = vec![0u32, 1, 2, 0];
        let mz = vec![100.000, 100.004, 200.000, 100.001];
        assert_eq!(fragment_cardinality(&cid, &mz), vec![2, 2, 1, 2]);
    }
}

#[cfg(test)]
mod im_tests {
    use super::*;

    #[test]
    fn im_requests_are_deduplicated_by_peptidoform_and_charge_in_first_seen_order() {
        let rows = [
            ("SAMPLER", 2, 401.2),
            ("PEPTIDEK", 2, 465.7),
            ("SAMPLER", 2, 401.2),
            ("SAMPLER", 3, 267.8),
            ("DECOY_KEDITPEP", 2, 465.7),
            ("PEPTIDEK", 2, 465.7),
        ];
        let keys = ImKeys::of(rows.iter().map(|&(p, z, mz)| (p, z, mz)));
        assert_eq!(
            keys.peps,
            ["SAMPLER", "PEPTIDEK", "SAMPLER", "DECOY_KEDITPEP"],
            "one request per (peptidoform, charge), the charge-3 form its own"
        );
        assert_eq!(keys.charges, [2, 2, 3, 2]);
        assert_eq!(keys.mzs, [401.2, 465.7, 267.8, 465.7]);
        assert_eq!(keys.row_ids, [0, 1, 0, 2, 3, 1]);
        assert_eq!(keys.ids(), [0, 1, 2, 3]);
        // The same rows give the same request, whatever the hasher.
        assert_eq!(ImKeys::of(rows.iter().map(|&(p, z, mz)| (p, z, mz))), keys);
    }

    #[test]
    fn im_values_follow_their_key_and_misses_are_reported_per_row() {
        let rows = [("A", 2, 1.0), ("B", 2, 2.0), ("A", 2, 1.0), ("A", 3, 0.7)];
        let keys = ImKeys::of(rows.iter().copied());
        // The worker returned nothing for ("A", 3).
        let out: HashMap<u32, f64> = [(0, 0.81), (1, 0.93)].into_iter().collect();
        let mut slots = vec![None; rows.len()];
        let missing = assign_im(slots.iter_mut(), &keys, &out);
        assert_eq!(slots, [Some(0.81), Some(0.93), Some(0.81), None]);
        assert_eq!(missing, [3]);
    }

    #[test]
    fn an_im_miss_is_dropped_with_its_pair_like_any_other() {
        // Rows 0 and 1 are a target and its decoy; row 2 another precursor. IM2Deep missed
        // the decoy only, and both members of the pair leave the library.
        let keys = vec![
            pair_key(7, 2, "PEPTIDEK"),
            pair_key(7, 2, "KEDITPEP"),
            pair_key(8, 2, "SAMPLER"),
        ];
        let im_missing = [1usize];
        let missing = unpredicted_rows(&[&[], &[], &im_missing]);
        assert_eq!(missing, [1]);
        assert_eq!(rows_to_drop(&keys, &missing), [true, true, false]);
        // One row missed by two predictors counts once.
        assert_eq!(unpredicted_rows(&[&[2, 0], &[2], &[1]]), [0, 1, 2]);
    }

    #[test]
    fn a_library_without_an_im_predictor_is_the_version_1_table() {
        let dir = std::env::temp_dir().join(format!("mumdia_pf_v1_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |n: &str| dir.join(n).to_string_lossy().into_owned();
        mumdia_io::table::write_table(
            &path("peptidoforms.parquet"),
            vec![
                Col::U32("id".into(), vec![0, 1]),
                Col::U32("base_peptide_id".into(), vec![0, 0]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEPTIDEK".into(), "KEDITPEP".into()],
                ),
                Col::I32("charge".into(), vec![2, 2]),
                Col::Str("label".into(), vec!["target".into(), "decoy".into()]),
                Col::Str("protein".into(), vec!["P1".into(), "DECOY_P1".into()]),
            ],
        )
        .unwrap();
        let cfg = PredictFragConfig::default();
        let (prec, out_f) = (path("prec.parquet"), path("frag.parquet"));
        run(PredictFragParams {
            peptidoforms: &path("peptidoforms.parquet"),
            out_precursors: &prec,
            out_fragments: &out_f,
            cfg: &cfg,
            work_dir: &path("work"),
            config_hash: "h",
            rt_placeholder: false,
        })
        .unwrap();
        assert_eq!(
            mumdia_io::table::column_names(&prec).unwrap(),
            [
                "candidate_id",
                "peptidoform_id",
                "base_peptide_id",
                "peptidoform",
                "charge",
                "precursor_mz",
                "predicted_irt",
                "label",
                "protein",
                "n_fragments"
            ]
        );
        let report: serde_json::Value =
            mumdia_io::json::read_json(&format!("{prec}.report.json")).unwrap();
        assert_eq!(report["schema_version"], 1);
        assert!(report["params"].get("im_predictor").is_none());
        assert!(report["stats"].get("without_im").is_none());
        assert!(!report["model_identity"]
            .as_str()
            .unwrap()
            .contains("im2deep"));
        assert_eq!(
            artifact::fragment_library_precursors(false),
            artifact::FRAGMENT_LIBRARY_PRECURSORS
        );
        assert_eq!(artifact::fragment_library_precursors(true).1, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
