//! Second pass on the pass-1 identifications (`rt_im_train.refit`, TIMS roadmap part 2,
//! L1d). The orchestrators run pass 1 into `<out>/pass1/`, then call [`run`], which
//! turns the accepted targets into a pseudo-seed, refits the RT library (cross-fitted
//! multi-head, when multi-head calibration is active) and rt-im-train on it, and writes
//! the pass-2 `run_windows.parquet`: the refit centres with the pass-1 half-widths.
//! Extract, features (with the ORIGINAL seed), compete and rescore then run again.

use std::sync::Arc;

use anyhow::{bail, Context as _, Result};
use arrow::array::{Array, ArrayRef, Float32Array};
use mumdia_core::config::Config;
use mumdia_io::table::{write_batches, write_table, Col, Table, TableFile};
use tracing::info;

use crate::stages::rt_im_train;

pub struct RefitParams<'a> {
    pub cfg: &'a Config,
    pub config_hash: &'a str,
    /// Pass-1 scored table of THIS run.
    pub scored: &'a str,
    /// PSM q column that selects the pseudo-seed: `q_value` in a single run,
    /// `run_psm_q` for one run of a pooled experiment.
    pub q_column: &'a str,
    /// Pass-1 extracted table (for `apex_im` at the selected peak).
    pub extracted: &'a str,
    /// Library pass 1 extracted with (its `predicted_irt` fills the pseudo-seed column).
    pub lib_pass1: &'a str,
    /// Library the multi-head calibration starts from (the pre-adaptation table).
    pub lib_base: &'a str,
    pub windows_pass1: &'a str,
    pub out_dir: &'a str,
    /// Multi-head heads; 0 keeps `lib_pass1` for pass 2.
    pub mh_heads: usize,
    /// A library an earlier run of the experiment already refitted
    /// (`experiment.rt_library_scope = first_run_only`).
    pub shared_lib: Option<&'a str>,
}

pub struct RefitOut {
    /// Library pass 2 extracts with.
    pub lib: String,
    /// Pass-2 windows (refit centres, pass-1 half-widths).
    pub windows: String,
    /// Set when this call produced a refitted library another run may reuse.
    pub produced_lib: Option<String>,
}

pub fn run(p: RefitParams) -> Result<RefitOut> {
    let d = |name: &str| format!("{}/{}", p.out_dir, name);
    let seed = d("pass1_seed.parquet");
    let n = pseudo_seed(
        p.scored,
        p.q_column,
        p.cfg.rt_im_train.q_train,
        p.extracted,
        p.lib_pass1,
        &seed,
    )?;
    info!(
        rows = n,
        "im-rt-refit: pseudo-seed from pass-1 accepted targets"
    );

    let (lib, produced_lib) = if let Some(shared) = p.shared_lib {
        (shared.to_string(), None)
    } else if p.mh_heads > 0 {
        let python = p
            .cfg
            .predict_frag
            .deeplc_python
            .as_deref()
            .expect("mh_heads is 0 unless an interpreter resolved");
        let script = crate::sidecar::resolve_script(
            &p.cfg.predict_frag.sidecar_script_dir,
            "deeplc_finetune.py",
        );
        let lib_ids = TableFile::open(p.lib_base)?.u32("base_peptide_id")?;
        let seed_fold = fold_of(&lib_ids, &TableFile::open(&seed)?.u32("base_peptide_id")?)?;
        let folds = fold_of_rows(&lib_ids);
        let mut fold_libs = Vec::new();
        for k in 0..2u8 {
            let s = d(&format!("pass1_seed_f{k}.parquet"));
            filter_rows(&seed, &s, |i| seed_fold[i] == k)?;
            let out = d(&format!("fragment_library_precursors_refit_f{k}.parquet"));
            info!(fold = k, "im-rt-refit: multi-head calibration on one fold");
            crate::sidecar::run_deeplc_multihead(
                python,
                &script,
                p.lib_base,
                &s,
                &out,
                p.mh_heads,
                p.cfg.rt_im_train.q_train,
                p.cfg.rt_im_train.window_holdout_frac,
                rayon::current_num_threads(),
                p.cfg.rt_im_train.deeplc_predict_shards,
                crate::cache::projection_dir(p.cfg).as_ref(),
            )?;
            fold_libs.push(out);
        }
        let out = d("fragment_library_precursors_refit.parquet");
        crossfit_merge(&fold_libs[0], &fold_libs[1], &folds, &out)?;
        (out.clone(), Some(out))
    } else {
        (p.lib_pass1.to_string(), None)
    };

    let refit_windows = d("run_windows_refit.parquet");
    rt_im_train::run(rt_im_train::RtImTrainParams {
        precursor_span: None,
        anchor_irt_from_seed: false,
        seed_psms: &seed,
        library_precursors: &lib,
        out_windows: &refit_windows,
        out_cal: &d("cal_refit.json"),
        cfg: &p.cfg.rt_im_train,
        config_hash: p.config_hash,
    })?;
    let windows = d("run_windows.parquet");
    recentre_windows(p.windows_pass1, &refit_windows, &windows)?;
    Ok(RefitOut {
        lib,
        windows,
        produced_lib,
    })
}

/// Pseudo-seed in the seed v2 schema: the targets with `q <= q_train` (the multi-head worker's rule; rt-im-train drops `q == q_train` itself), `apex_rt` as
/// `observed_rt`, extract's `apex_im` at the selected peak as `observed_im`, and the q
/// value as `spectrum_q`. Sorted by `candidate_id`.
fn pseudo_seed(
    scored: &str,
    q_column: &str,
    q_train: f64,
    extracted: &str,
    lib: &str,
    out: &str,
) -> Result<u64> {
    let s = TableFile::open(scored)?;
    let q = s.f64(q_column)?;
    let is_target = s.str_eq("label", "target")?;
    let keep: Vec<usize> = (0..s.nrows)
        .filter(|&i| is_target[i] && q[i].is_finite() && q[i] <= q_train)
        .collect();
    let cid = s.u32("candidate_id")?;
    let rank = s.i32("selected_peak_rank")?;

    // apex_im per (candidate, peak rank); ordered map, so nothing depends on hash order.
    let e = TableFile::open(extracted)?;
    let (e_cid, e_rank) = (e.u32("candidate_id")?, e.i32("peak_rank")?);
    let e_im = if e.has_column("apex_im") {
        e.opt_f64("apex_im")?
    } else {
        vec![None; e.nrows]
    };
    let im: std::collections::BTreeMap<(u32, i32), Option<f64>> = (0..e.nrows)
        .map(|i| ((e_cid[i], e_rank[i]), e_im[i].filter(|v| v.is_finite())))
        .collect();

    let l = TableFile::open(lib)?;
    let l_cid = l.u32("candidate_id")?;
    let (l_mz, l_irt) = (l.f64("precursor_mz")?, l.f32("predicted_irt")?);
    // candidate_id is the contiguous row index of a library (index.rs), but look it up
    // rather than trust that here.
    let lib_row: std::collections::BTreeMap<u32, usize> =
        l_cid.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    let mut keep = keep;
    keep.sort_by_key(|&i| cid[i]);
    let row = |i: usize| -> Result<usize> {
        lib_row
            .get(&cid[i])
            .copied()
            .with_context(|| format!("candidate {} is not in {lib}", cid[i]))
    };
    let lr: Vec<usize> = keep.iter().map(|&i| row(i)).collect::<Result<_>>()?;
    let pick_str = |v: Vec<String>| keep.iter().map(|&i| v[i].clone()).collect::<Vec<_>>();
    let (pform, prot, label) = (
        pick_str(s.str("peptidoform")?),
        pick_str(s.str("protein")?),
        pick_str(s.str("label")?),
    );
    let (charge, base, score, rt) = (
        s.i32("charge")?,
        s.u32("base_peptide_id")?,
        s.f64("score")?,
        s.f64("apex_rt")?,
    );
    let n = keep.len();
    write_table(
        out,
        vec![
            Col::U32(
                "candidate_id".into(),
                keep.iter().map(|&i| cid[i]).collect(),
            ),
            Col::Str("peptidoform".into(), pform),
            Col::I32("charge".into(), keep.iter().map(|&i| charge[i]).collect()),
            Col::F64("precursor_mz".into(), lr.iter().map(|&r| l_mz[r]).collect()),
            Col::U32(
                "base_peptide_id".into(),
                keep.iter().map(|&i| base[i]).collect(),
            ),
            Col::Str("protein".into(), prot),
            Col::Str("label".into(), label),
            Col::F64("score".into(), keep.iter().map(|&i| score[i]).collect()),
            Col::F64("spectrum_q".into(), keep.iter().map(|&i| q[i]).collect()),
            Col::F64("observed_rt".into(), keep.iter().map(|&i| rt[i]).collect()),
            Col::F32(
                "predicted_irt".into(),
                lr.iter().map(|&r| l_irt[r]).collect(),
            ),
            Col::I32("matched_peaks".into(), vec![0; n]),
            Col::U32("scan_index".into(), vec![0; n]),
            Col::OptF64(
                "observed_im".into(),
                keep.iter()
                    .map(|&i| im.get(&(cid[i], rank[i])).copied().flatten())
                    .collect(),
            ),
        ],
    )
}

/// Fold (0 or 1) of each id: the parity of its rank among the distinct ids of the
/// library. Target/decoy pairs share `base_peptide_id` and so a fold. On a native digest
/// (ids 0, 2, 4, ...) this is `(id >> 1) & 1`; on an imported library (0, 1, 2, ...) it
/// is the id parity.
fn fold_of(lib_ids: &[u32], ids: &[u32]) -> Result<Vec<u8>> {
    let mut u = lib_ids.to_vec();
    u.sort_unstable();
    u.dedup();
    ids.iter()
        .map(|id| {
            u.binary_search(id)
                .map(|r| (r & 1) as u8)
                .map_err(|_| anyhow::anyhow!("base_peptide_id {id} is not in the library"))
        })
        .collect()
}

fn fold_of_rows(lib_ids: &[u32]) -> Vec<u8> {
    fold_of(lib_ids, lib_ids).expect("every id is in its own library")
}

fn filter_rows(src: &str, out: &str, keep: impl Fn(usize) -> bool) -> Result<u64> {
    let t = Table::read(src)?;
    let mut off = 0;
    let mut batches = Vec::with_capacity(t.batches.len());
    for b in &t.batches {
        let mask: arrow::array::BooleanArray =
            (0..b.num_rows()).map(|i| Some(keep(off + i))).collect();
        off += b.num_rows();
        batches.push(arrow::compute::filter_record_batch(b, &mask)?);
    }
    write_batches(out, t.schema.clone(), &batches)
}

/// The fold-0 library with each row's `predicted_irt` taken from the fit on the OTHER
/// fold: rows of fold 0 from `lib_f1`, rows of fold 1 from `lib_f0`. No anchor is then
/// predicted by a model that saw it, and targets and decoys are treated alike.
fn crossfit_merge(lib_f0: &str, lib_f1: &str, folds: &[u8], out: &str) -> Result<u64> {
    let a = Table::read(lib_f0)?;
    let b = Table::read_cols(lib_f1, &["candidate_id", "predicted_irt"])?;
    if a.u32("candidate_id")? != b.u32("candidate_id")? || folds.len() != a.nrows {
        bail!("{lib_f0} and {lib_f1} are not row-aligned with the base library");
    }
    let mut irt_b: Vec<Option<f32>> = Vec::with_capacity(b.nrows);
    let bc = b.schema.index_of("predicted_irt")?;
    for batch in &b.batches {
        let v = batch
            .column(bc)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("predicted_irt is not float32")?;
        irt_b.extend((0..v.len()).map(|i| v.is_valid(i).then(|| v.value(i))));
    }
    let col = a
        .schema
        .index_of("predicted_irt")
        .context("predicted_irt missing from the multi-head library")?;
    let mut off = 0;
    let mut batches = Vec::with_capacity(a.batches.len());
    for batch in &a.batches {
        let irt_a = batch
            .column(col)
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("predicted_irt is not float32")?;
        let v: Float32Array = (0..batch.num_rows())
            .map(|i| {
                if folds[off + i] == 0 {
                    irt_b[off + i]
                } else {
                    irt_a.is_valid(i).then(|| irt_a.value(i))
                }
            })
            .collect();
        off += batch.num_rows();
        let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
        cols[col] = Arc::new(v);
        batches.push(arrow::record_batch::RecordBatch::try_new(
            batch.schema(),
            cols,
        )?);
    }
    write_batches(out, a.schema.clone(), &batches)
}

/// Refit centres with the pass-1 half-widths, per candidate and per axis. The refit
/// widths are truncated by selection (pass-1 IDs can only lie inside the pass-1
/// windows) and measured 4% fewer peptides.
fn recentre_windows(old: &str, refit: &str, out: &str) -> Result<u64> {
    let (o, n) = (TableFile::open(old)?, TableFile::open(refit)?);
    let cid = n.u32("candidate_id")?;
    if o.u32("candidate_id")? != cid {
        bail!("{old} and {refit} are not row-aligned");
    }
    let (op, olo, ohi, nc) = (
        o.f64("rt_pred_cal")?,
        o.f64("rt_lo")?,
        o.f64("rt_hi")?,
        n.f64("rt_pred_cal")?,
    );
    let rt_lo = (0..cid.len()).map(|i| nc[i] - (op[i] - olo[i])).collect();
    let rt_hi = (0..cid.len()).map(|i| nc[i] + (ohi[i] - op[i])).collect();
    let (oip, oilo, oihi, nic) = (
        o.opt_f64("im_pred_cal")?,
        o.opt_f64("im_lo")?,
        o.opt_f64("im_hi")?,
        n.opt_f64("im_pred_cal")?,
    );
    let im = |f: &dyn Fn(f64, f64, f64, f64) -> f64| -> Vec<Option<f64>> {
        (0..cid.len())
            .map(|i| Some(f(nic[i]?, oip[i]?, oilo[i]?, oihi[i]?)))
            .collect()
    };
    let im_lo = im(&|c, p, lo, _| c - (p - lo));
    let im_hi = im(&|c, p, _, hi| c + (hi - p));
    write_table(
        out,
        vec![
            Col::U32("candidate_id".into(), cid),
            Col::F64("rt_pred_cal".into(), nc),
            Col::F64("rt_lo".into(), rt_lo),
            Col::F64("rt_hi".into(), rt_hi),
            Col::OptF64("im_pred_cal".into(), nic),
            Col::OptF64("im_lo".into(), im_lo),
            Col::OptF64("im_hi".into(), im_hi),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_alternate_over_distinct_ids_on_both_numberings() {
        // native digest: steps of 2, equal to (id >> 1) & 1
        let dig = [0u32, 0, 2, 4, 6, 6];
        assert_eq!(fold_of_rows(&dig), vec![0, 0, 1, 0, 1, 1]);
        for &id in &dig {
            assert_eq!(fold_of(&dig, &[id]).unwrap()[0] as u32, (id >> 1) & 1);
        }
        // imported: steps of 1, the parity
        assert_eq!(fold_of_rows(&[3, 0, 1, 2]), vec![1, 0, 1, 0]);
        assert!(fold_of(&dig, &[5]).is_err());
    }

    #[test]
    fn windows_keep_old_widths_around_new_centres() {
        let dir = std::env::temp_dir().join(format!("imrt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = |n: &str| dir.join(n).to_string_lossy().into_owned();
        let w = |path: &str, c: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>, im: Vec<Option<f64>>| {
            write_table(
                path,
                vec![
                    Col::U32("candidate_id".into(), vec![0, 1]),
                    Col::F64("rt_pred_cal".into(), c),
                    Col::F64("rt_lo".into(), lo),
                    Col::F64("rt_hi".into(), hi),
                    Col::OptF64("im_pred_cal".into(), im.clone()),
                    Col::OptF64(
                        "im_lo".into(),
                        im.iter().map(|v| v.map(|x| x - 0.1)).collect(),
                    ),
                    Col::OptF64(
                        "im_hi".into(),
                        im.iter().map(|v| v.map(|x| x + 0.2)).collect(),
                    ),
                ],
            )
            .unwrap();
        };
        w(
            &p("o"),
            vec![100.0, 200.0],
            vec![90.0, 180.0],
            vec![120.0, 230.0],
            vec![Some(1.0), None],
        );
        w(
            &p("n"),
            vec![105.0, 190.0],
            vec![104.0, 189.0],
            vec![106.0, 191.0],
            vec![Some(1.5), Some(0.9)],
        );
        recentre_windows(&p("o"), &p("n"), &p("r")).unwrap();
        let r = TableFile::open(&p("r")).unwrap();
        assert_eq!(r.f64("rt_lo").unwrap(), vec![95.0, 170.0]);
        assert_eq!(r.f64("rt_hi").unwrap(), vec![125.0, 220.0]);
        let lo = r.opt_f64("im_lo").unwrap();
        assert!((lo[0].unwrap() - 1.4).abs() < 1e-12 && lo[1].is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
