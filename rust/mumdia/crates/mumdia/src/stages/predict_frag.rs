//! Stage C `mumdia predict-frag`: spectral library generation (PLAN.md Stage C).
//! Experiment-wide and run-independent. Computes precursor and b/y fragment m/z
//! from the UniMod-backed mass model, predicted fragment intensities and iRT
//! (native or MS2PIP/DeepLC sidecars over the file contract), keeps top-N
//! fragments, and assigns `candidate_id` as the precursor-m/z sort key so the
//! extractor's index build is a contiguous range (PLAN.md Stage D build step 1).

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{bail, Result};
use mumdia_core::config::{FragPredictorKind, PredictFragConfig, RtPredictorKind};
use mumdia_core::mass::{parse_peptidoform, Fragment, ParsedPeptidoform};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{write_table, Col, Table};
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
    frag_int: Vec<f32>,
    /// Parsed form kept from phase A so `assign_rt`/`assign_intensities` reuse
    /// it instead of re-parsing `peptidoform` (identical parse output).
    parsed: ParsedPeptidoform,
}

pub fn run(p: PredictFragParams) -> Result<(u64, u64)> {
    let t0 = Instant::now();
    let t = Table::read(p.peptidoforms)?;
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
            let mut frag_charges = vec![1];
            if z >= p.cfg.charge2_from_precursor_charge {
                frag_charges.push(2);
            }
            let frags = parsed.fragments(&frag_charges);
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
    info!(candidates = raws.len(), parse_errors = n_parse_err, "predict-frag: parsed");

    let rt_model_id = assign_rt(&p, &mut raws)?;
    let frag_model_id = assign_intensities(&p, &mut raws)?;
    let model_identity = format!("{rt_model_id}; {frag_model_id}");

    // Keep top-N fragments by predicted intensity. Each candidate is independent
    // (operates only on its own frags/frag_int), so this parallelizes with no
    // cross-item state; the result per candidate is identical to the serial loop.
    raws.par_iter_mut().for_each(|r| {
        let mut order: Vec<usize> = (0..r.frags.len()).collect();
        order.sort_by(|&a, &b| r.frag_int[b].partial_cmp(&r.frag_int[a]).unwrap());
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

    // candidate_id = precursor-m/z sort key (PLAN.md Stage D build step 1).
    raws.sort_by(|a, b| a.precursor_mz.partial_cmp(&b.precursor_mz).unwrap());

    let n = raws.len();
    let total_frags: usize = raws.iter().map(|r| r.frags.len()).sum();
    let (mut cid, mut pfid, mut baseid) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut pform_c, mut z_c, mut mz_c, mut irt_c) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut label_c, mut prot_c, mut nfrag_c) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
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
        for (fr, fi) in r.frags.into_iter().zip(r.frag_int) {
            f_cid.push(candidate_id);
            f_mz.push(fr.mz);
            f_int.push(fi);
            f_name.push(fr.name);
            f_type.push(fr.ion_type.symbol().to_string());
            f_ord.push(fr.ordinal as i32);
            f_chg.push(fr.charge);
        }
    }

    let n_prec = write_table(
        p.out_precursors,
        vec![
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
        ],
    )?;
    let n_frag = write_table(
        p.out_fragments,
        vec![
            Col::U32("candidate_id".into(), f_cid),
            Col::F64("mz".into(), f_mz),
            Col::F32("predicted_intensity".into(), f_int),
            Col::Str("name".into(), f_name),
            Col::Str("ion_type".into(), f_type),
            Col::I32("ordinal".into(), f_ord),
            Col::I32("frag_charge".into(), f_chg),
        ],
    )?;

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("candidates".to_string(), json!(n_prec));
    stats.insert("fragments".to_string(), json!(n_frag));
    stats.insert("parse_errors".to_string(), json!(n_parse_err));
    for (path, schema) in [
        (p.out_precursors, artifact::FRAGMENT_LIBRARY_PRECURSORS),
        (p.out_fragments, artifact::FRAGMENT_LIBRARY_FRAGMENTS),
    ] {
        ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "predict-frag".to_string(),
            rows: if path == p.out_precursors { n_prec } else { n_frag },
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: json!({"top_n": p.cfg.top_n_fragments, "ms2pip_model": p.cfg.ms2pip_model,
                           "rt_predictor": format!("{:?}", p.cfg.rt_predictor),
                           "fragment_predictor": format!("{:?}", p.cfg.predictor)}),
            stats: stats.clone(),
            model_identity: Some(model_identity.clone()),
            elapsed_ms: elapsed,
        }
        .write_for(path)?;
    }

    info!(candidates = n_prec, fragments = n_frag, elapsed_ms = elapsed, "predict-frag: done");
    Ok((n_prec, n_frag))
}

/// Assign predicted iRT to every candidate. Returns the model id.
fn assign_rt(p: &PredictFragParams, raws: &mut [Raw]) -> Result<String> {
    match p.cfg.rt_predictor {
        RtPredictorKind::Native => {
            let m = NativeRt;
            for r in raws.iter_mut() {
                r.irt = m.predict_irt(&r.parsed);
            }
            Ok(m.identity())
        }
        RtPredictorKind::Deeplc => {
            let python = p
                .cfg
                .deeplc_python
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("rt_predictor=deeplc requires predict_frag.deeplc_python"))?;
            let script = crate::sidecar::resolve_script(&p.cfg.sidecar_script_dir, "deeplc_worker.py");
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
            let out = sidecar::run_deeplc(python, &script, p.work_dir, &ids, &peps)?;
            let mut n_irt_missing = 0u64;
            for r in raws.iter_mut() {
                let uid = uniq[&r.peptidoform];
                match out.get(&uid) {
                    Some(&v) => r.irt = v,
                    None => {
                        r.irt = 0.0;
                        n_irt_missing += 1;
                    }
                }
            }
            if n_irt_missing > 0 {
                tracing::warn!(
                    n_irt_missing,
                    "predict-frag: DeepLC returned no iRT for some peptidoforms; anchored at iRT 0.0"
                );
            }
            Ok("deeplc-4.0-mt".to_string())
        }
    }
}

/// Assign a predicted intensity to every fragment. Returns the model id.
fn assign_intensities(p: &PredictFragParams, raws: &mut [Raw]) -> Result<String> {
    match p.cfg.predictor {
        FragPredictorKind::Native => {
            let m = NativeFrag;
            for r in raws.iter_mut() {
                r.frag_int = m.predict_intensities(&r.parsed, &r.frags);
            }
            Ok(m.identity())
        }
        FragPredictorKind::Ms2pip => {
            let python = p
                .cfg
                .ms2pip_python
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("predictor=ms2pip requires predict_frag.ms2pip_python"))?;
            let script = crate::sidecar::resolve_script(&p.cfg.sidecar_script_dir, "ms2pip_worker.py");
            let ids: Vec<u32> = (0..raws.len() as u32).collect();
            let peps: Vec<String> = raws.iter().map(|r| r.peptidoform.clone()).collect();
            let charges: Vec<i32> = raws.iter().map(|r| r.charge).collect();
            let map = sidecar::run_ms2pip(python, &script, p.work_dir, &ids, &peps, &charges, &p.cfg.ms2pip_model)?;
            if map.is_empty() {
                bail!("MS2PIP returned no predictions");
            }
            let native = NativeFrag;
            for (i, r) in raws.iter_mut().enumerate() {
                let per = map.get(&(i as u32));
                match per {
                    Some(per) if !per.is_empty() => {
                        // native as fallback for fragment charges MS2PIP does not emit (charge 2)
                        let nat = native.predict_intensities(&r.parsed, &r.frags);
                        let mut vals: Vec<f32> = r
                            .frags
                            .iter()
                            .enumerate()
                            .map(|(k, fr)| {
                                if fr.charge == 1 {
                                    let ion = fr.ion_type.symbol() as u8;
                                    *per.get(&(ion, fr.ordinal as u16)).unwrap_or(&0.0)
                                } else {
                                    nat[k]
                                }
                            })
                            .collect();
                        // MS2PIP (charge-1, TIC-fraction, ~0.02-0.3) and the native
                        // charge-2 fallback (max-normalized, ~0.19-0.5) live on
                        // different scales; ranking them together in top-N buries
                        // MS2PIP. Max-normalize each charge group to its own peak so
                        // the two compete fairly.
                        let gmax = |want2: bool| {
                            r.frags
                                .iter()
                                .zip(&vals)
                                .filter(|(fr, _)| (fr.charge >= 2) == want2)
                                .map(|(_, v)| *v)
                                .fold(0.0f32, f32::max)
                        };
                        let (m1, m2) = (gmax(false), gmax(true));
                        for (k, fr) in r.frags.iter().enumerate() {
                            let m = if fr.charge >= 2 { m2 } else { m1 };
                            if m > 0.0 {
                                vals[k] /= m;
                            }
                        }
                        r.frag_int = vals;
                    }
                    _ => {
                        r.frag_int = native.predict_intensities(&r.parsed, &r.frags);
                    }
                }
            }
            Ok(format!("ms2pip-{}", p.cfg.ms2pip_model))
        }
    }
}
