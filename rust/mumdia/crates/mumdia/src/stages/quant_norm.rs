//! Cross-run normalisation of per-run precursor quantities (`quant.normalisation =
//! "stable_set"`, docs/12_quant_lfq_align_mbr_report_audit.md, "Cross-run normalisation").
//!
//! `peptide_quant.quantity` is a raw area, so a run that was loaded or sprayed more carries
//! larger quantities for every precursor, and the deviation depends on retention time. This
//! module estimates one log2 shift per run as a function of retention time with the
//! stable-set estimator of [`super::quant_stable`] and writes it, with the shifted quantity,
//! to a separate table per run. `peptide_quant` is not changed.
//!
//! Ions are indexed in the order the method was confirmed in: run by run, each run's rows
//! by `candidate_id` (row order without the column), every row's key at its first
//! appearance. A run's duplicate rows of one ion keep the largest quantity, the last of
//! equal ones. `quantity_normalised = quantity / 2^shift`.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{bail, Context as _, Result};
use mumdia_core::config::{CrossRunNormalisation, QuantConfig};
use mumdia_core::schema::artifact;
use mumdia_io::report::{ArtifactReport, Written};
use mumdia_io::table::{write_table_hashed, Col, TableFile};
use serde_json::json;
use tracing::{info, warn};

use super::quant_stable::{self, StableFit};

/// The number of runs an ion must be quantified in to enter the estimate, its own run
/// included.
pub fn min_runs(n_runs: usize) -> usize {
    n_runs.div_ceil(2).max(2)
}

/// Inputs and outputs of [`run`].
pub struct NormaliseParams<'a> {
    /// One `peptide_quant.parquet` per run, in run order.
    pub inputs: &'a [String],
    /// One `peptide_quant_normalised.parquet` per run, in the same order.
    pub outputs: &'a [String],
    /// The shifts table (`quant_normalisation.parquet`).
    pub shifts_out: &'a str,
    pub cfg: &'a QuantConfig,
    pub config_hash: &'a str,
}

/// What [`run`] wrote: each run's normalised table, then the shifts table.
pub struct NormaliseWritten {
    pub tables: Vec<Written>,
    pub shifts: Written,
    pub n_shift_rows: u64,
}

/// One run's `peptide_quant` columns.
struct QuantTable {
    /// `candidate_id`, when the table has it: the row order the estimator indexes ions in.
    cid: Option<Vec<u32>>,
    pform: Vec<String>,
    charge: Vec<i32>,
    protein_group: Vec<String>,
    quantity: Vec<Option<f64>>,
    status: Vec<String>,
    n_fragments: Vec<i32>,
    apex_rt: Vec<Option<f64>>,
}

fn read_quant(path: &str) -> Result<QuantTable> {
    let t = TableFile::open(path).with_context(|| format!("opening {path}"))?;
    let apex_rt = if t.has_column("integration_apex_rt") {
        t.opt_f64("integration_apex_rt")?
    } else {
        vec![None; t.nrows]
    };
    let cid = if t.has_column("candidate_id") {
        Some(t.u32("candidate_id")?)
    } else {
        None
    };
    Ok(QuantTable {
        cid,
        pform: t.str("peptidoform")?,
        charge: t.i32("charge")?,
        protein_group: t.str("protein_group")?,
        quantity: t.opt_f64("quantity")?,
        status: t.str("quant_status")?,
        n_fragments: t.i32("n_fragments_used")?,
        apex_rt,
    })
}

fn positive(q: Option<f64>) -> Option<f64> {
    q.filter(|v| v.is_finite() && *v > 0.0)
}

/// Normalise every run's `peptide_quant` and write the normalised tables and the shifts.
///
/// The normalised table of a run is row-aligned with its `peptide_quant`: the key columns,
/// `quant_status`, `n_fragments_used` and `integration_apex_rt` copied, the raw `quantity`,
/// `quantity_normalised = quantity / 2^normalisation_log2_shift` and the shift at the row's
/// apex. A row without a positive finite quantity keeps its value in both columns.
pub fn run(p: NormaliseParams) -> Result<NormaliseWritten> {
    let t0 = Instant::now();
    if p.inputs.len() != p.outputs.len() {
        bail!(
            "quant-normalise: {} inputs but {} outputs",
            p.inputs.len(),
            p.outputs.len()
        );
    }
    if p.inputs.is_empty() {
        bail!("quant-normalise: no input tables");
    }
    let n = p.inputs.len();
    if n < 2 {
        warn!("quant-normalise: one run has no cross-run reference; its shift is 0");
    }
    if let Some(k) = p.cfg.normalisation_rank {
        if n >= 2 && k + 2 > n {
            bail!(
                "quant.normalisation_rank = {k} leaves no residual dimension with {n} runs; it \
                 must be at most {} (the number of sample groups less one)",
                n.saturating_sub(2)
            );
        }
    }
    let tables: Vec<QuantTable> = p
        .inputs
        .iter()
        .map(|path| read_quant(path))
        .collect::<Result<_>>()?;
    let order_of = |t: &QuantTable| -> Vec<usize> {
        let mut o: Vec<usize> = (0..t.pform.len()).collect();
        if let Some(c) = &t.cid {
            o.sort_by_key(|&i| c[i]);
        }
        o
    };
    let orders: Vec<Vec<usize>> = tables.iter().map(order_of).collect();
    let mut ion_index: BTreeMap<(&str, i32), usize> = BTreeMap::new();
    for (t, o) in tables.iter().zip(&orders) {
        for &i in o {
            let next = ion_index.len();
            ion_index
                .entry((t.pform[i].as_str(), t.charge[i]))
                .or_insert(next);
        }
    }
    let n_ions = ion_index.len();
    let mut y = vec![f64::NAN; n_ions * n];
    let mut rt = vec![f64::NAN; n_ions * n];
    for (r, (t, o)) in tables.iter().zip(&orders).enumerate() {
        for &i in o {
            let Some(q) = positive(t.quantity[i]) else {
                continue;
            };
            let ion = ion_index[&(t.pform[i].as_str(), t.charge[i])];
            let v = q.log2();
            let cell = ion * n + r;
            if y[cell].is_nan() || v >= y[cell] {
                y[cell] = v;
                rt[cell] = t.apex_rt[i].filter(|x| x.is_finite()).unwrap_or(f64::NAN);
            }
        }
    }
    let fit = quant_stable::fit(
        &quant_stable::Matrix {
            y: &y,
            rt: &rt,
            n_ions,
            runs: n,
        },
        p.cfg.normalisation_rank,
    );
    for (pass, info) in fit.passes.iter().enumerate() {
        info!(
            pass = pass + 1,
            k = info.k,
            complete_ions = info.n_complete,
            stable_ions = info.n_stable,
            sigma = info.sigma,
            "quant-normalise: stable set"
        );
    }
    write_outputs(&p, &tables, &fit, n_ions, t0)
}

/// Write each run's normalised table and the shifts table of `fit`.
fn write_outputs(
    p: &NormaliseParams,
    tables: &[QuantTable],
    fit: &StableFit,
    n_ions: usize,
    t0: Instant,
) -> Result<NormaliseWritten> {
    let n = tables.len();
    let elapsed = t0.elapsed().as_millis();
    // Fit ions of the last pass that estimated the shift; 0 when the run is not shifted.
    let n_estimation = |c: &quant_stable::Curve| if c.fitted { c.n_fit } else { 0 };
    let mut written: Vec<Written> = Vec::with_capacity(n);
    for (r, t) in tables.iter().enumerate() {
        let c = &fit.curves[r];
        let shift: Vec<f64> = t.apex_rt.iter().map(|&rt| c.at(rt)).collect();
        let normalised: Vec<Option<f64>> = t
            .quantity
            .iter()
            .zip(&shift)
            .map(|(&q, &d)| match positive(q) {
                Some(v) => Some(v / d.exp2()),
                None => q,
            })
            .collect();
        let out = &p.outputs[r];
        let w = write_table_hashed(
            out,
            vec![
                Col::Str("peptidoform".into(), t.pform.clone()),
                Col::I32("charge".into(), t.charge.clone()),
                Col::Str("protein_group".into(), t.protein_group.clone()),
                Col::OptF64("quantity".into(), t.quantity.clone()),
                Col::OptF64("quantity_normalised".into(), normalised),
                Col::F64("normalisation_log2_shift".into(), shift),
                Col::Str("quant_status".into(), t.status.clone()),
                Col::I32("n_fragments_used".into(), t.n_fragments.clone()),
                Col::OptF64("integration_apex_rt".into(), t.apex_rt.clone()),
            ],
        )?;
        let mut stats = BTreeMap::new();
        stats.insert("global_log2_shift".to_string(), json!(c.global));
        stats.insert("n_rt_bins".to_string(), json!(c.knot_rt.len()));
        stats.insert("n_ions".to_string(), json!(c.n_fit));
        stats.insert("n_estimation".to_string(), json!(n_estimation(c)));
        let report = ArtifactReport {
            logical_name: artifact::PEPTIDE_QUANT_NORMALISED.0.to_string(),
            schema_name: artifact::PEPTIDE_QUANT_NORMALISED.0.to_string(),
            schema_version: artifact::PEPTIDE_QUANT_NORMALISED.1,
            stage: "quant-normalise".to_string(),
            rows: w.rows,
            content_hash: w.content_hash.clone(),
            params: json!({ "config_hash": p.config_hash, "run": r, "input": p.inputs[r] }),
            stats,
            model_identity: None,
            elapsed_ms: elapsed,
        };
        report.write_for(out)?;
        if !c.fitted && n >= 2 {
            warn!(
                run = r,
                n_ions = c.n_fit,
                min_ions_per_bin = quant_stable::MIN_IONS,
                "quant-normalise: too few estimation ions; the run is not shifted"
            );
        }
        info!(
            run = r,
            global_log2_shift = c.global,
            rt_bins = c.knot_rt.len(),
            n_ions = c.n_fit,
            n_estimation = n_estimation(c),
            "quant-normalise: run shift"
        );
        written.push(w);
    }
    // The shifts: per run, one row for the global shift (bin -1, no retention time), then
    // one per retention-time bin.
    let (mut c_run, mut c_bin, mut c_lo, mut c_hi, mut c_ctr) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut c_n, mut c_ne, mut c_median, mut c_shift) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (r, c) in fit.curves.iter().enumerate() {
        c_run.push(r as i32);
        c_bin.push(-1);
        c_lo.push(None);
        c_hi.push(None);
        c_ctr.push(None);
        c_n.push(c.n_fit as i32);
        c_ne.push(n_estimation(c) as i32);
        c_median.push(c.global);
        c_shift.push(c.global);
        for b in 0..c.knot_rt.len() {
            c_run.push(r as i32);
            c_bin.push(b as i32);
            c_lo.push(Some(c.knot_lo[b]));
            c_hi.push(Some(c.knot_hi[b]));
            c_ctr.push(Some(c.knot_rt[b]));
            c_n.push(c.knot_n[b] as i32);
            c_ne.push(c.knot_n[b] as i32);
            c_median.push(c.knot_median[b]);
            c_shift.push(c.knot_shift[b]);
        }
    }
    let sw = write_table_hashed(
        p.shifts_out,
        vec![
            Col::I32("run".into(), c_run),
            Col::I32("bin".into(), c_bin),
            Col::OptF64("rt_lo".into(), c_lo),
            Col::OptF64("rt_hi".into(), c_hi),
            Col::OptF64("rt_center".into(), c_ctr),
            Col::I32("n_ions".into(), c_n),
            Col::I32("n_estimation".into(), c_ne),
            Col::F64("log2_median".into(), c_median),
            Col::F64("log2_shift".into(), c_shift),
        ],
    )?;
    let mut stats = BTreeMap::new();
    stats.insert("n_runs".to_string(), json!(n));
    stats.insert("n_ions".to_string(), json!(n_ions));
    stats.insert(
        "global_log2_shift".to_string(),
        json!(fit.curves.iter().map(|c| c.global).collect::<Vec<f64>>()),
    );
    stats.insert(
        "stable_set".to_string(),
        json!(fit
            .passes
            .iter()
            .map(|i| json!({
                "k": i.k,
                "eigenvalues": i.eigenvalues,
                "sigma": i.sigma,
                "complete_ions": i.n_complete,
                "stable_ions": i.n_stable,
            }))
            .collect::<Vec<_>>()),
    );
    let report = ArtifactReport {
        logical_name: artifact::QUANT_NORMALISATION.0.to_string(),
        schema_name: artifact::QUANT_NORMALISATION.0.to_string(),
        schema_version: artifact::QUANT_NORMALISATION.1,
        stage: "quant-normalise".to_string(),
        rows: sw.rows,
        content_hash: sw.content_hash.clone(),
        params: json!({
            "method": format!("{:?}", CrossRunNormalisation::StableSet),
            "rank": p.cfg.normalisation_rank.map_or(json!("auto"), |k| json!(k)),
            "rt_bins": quant_stable::RT_BINS,
            "min_ions_per_bin": quant_stable::MIN_IONS,
            "min_runs": min_runs(n),
            "iterations": quant_stable::ITERATIONS,
            "inputs": p.inputs,
            "config_hash": p.config_hash,
        }),
        stats,
        model_identity: None,
        elapsed_ms: elapsed,
    };
    report.write_for(p.shifts_out)?;
    let n_shift_rows = sw.rows;
    Ok(NormaliseWritten {
        tables: written,
        shifts: sw,
        n_shift_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(name: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir =
            std::env::temp_dir().join(format!("mumdia_quant_norm_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        dir.join(format!("{n}_{name}"))
            .to_string_lossy()
            .into_owned()
    }

    /// A deterministic pseudo-normal deviate (sum of 12 uniforms) from a 64-bit LCG.
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
        }
    }

    #[test]
    fn min_runs_is_half_and_at_least_two() {
        assert_eq!(min_runs(2), 2);
        assert_eq!(min_runs(3), 2);
        assert_eq!(min_runs(6), 3);
        assert_eq!(min_runs(18), 9);
    }

    fn write_quant(path: &str, rows: &[(&str, i32, Option<f64>, Option<f64>)]) {
        mumdia_io::table::write_table(
            path,
            vec![
                Col::Str(
                    "peptidoform".into(),
                    rows.iter().map(|r| r.0.to_string()).collect(),
                ),
                Col::I32("charge".into(), rows.iter().map(|r| r.1).collect()),
                Col::Str("protein_group".into(), vec!["PG".into(); rows.len()]),
                Col::OptF64("quantity".into(), rows.iter().map(|r| r.2).collect()),
                Col::Str("quant_status".into(), vec!["ok".into(); rows.len()]),
                Col::I32("n_fragments_used".into(), vec![3; rows.len()]),
                Col::OptF64(
                    "integration_apex_rt".into(),
                    rows.iter().map(|r| r.3).collect(),
                ),
            ],
        )
        .unwrap();
    }

    /// End to end on tables: a run loaded 2x higher is brought back, the output is
    /// row-aligned with the input, null or zero quantities pass through, and a second call
    /// writes the same bytes.
    #[test]
    fn run_writes_row_aligned_normalised_tables_and_the_shifts() {
        const N: usize = 1000;
        let names: Vec<String> = (0..N).map(|i| format!("PEP{i}")).collect();
        let mut rng = Rng(9);
        let base: Vec<f64> = (0..N)
            .map(|_| 2f64.powf(15.0 + 5.0 * rng.uniform()))
            .collect();
        let inputs: Vec<String> = (0..6)
            .map(|r| test_path(&format!("pq{r}.parquet")))
            .collect();
        let outputs: Vec<String> = (0..6)
            .map(|r| test_path(&format!("pqn{r}.parquet")))
            .collect();
        for (r, path) in inputs.iter().enumerate() {
            let load = if r == 5 { 2.0 } else { 1.0 };
            let mut rows: Vec<(&str, i32, Option<f64>, Option<f64>)> = names
                .iter()
                .zip(&base)
                .enumerate()
                .map(|(i, (n, &q))| {
                    let noise = (0.02 * rng.normal()).exp2();
                    (n.as_str(), 2, Some(q * load * noise), Some(i as f64))
                })
                .collect();
            rows.push(("NULLPEP", 2, None, Some(1.0)));
            rows.push(("ZEROPEP", 3, Some(0.0), None));
            write_quant(path, &rows);
        }
        let cfg = QuantConfig {
            normalisation: CrossRunNormalisation::StableSet,
            ..QuantConfig::default()
        };
        let shifts_out = test_path("shifts.parquet");
        let w = run(NormaliseParams {
            inputs: &inputs,
            outputs: &outputs,
            shifts_out: &shifts_out,
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        assert_eq!(w.tables.len(), 6);
        for (r, out) in outputs.iter().enumerate() {
            let t = TableFile::open(out).unwrap();
            assert_eq!(t.nrows, N + 2);
            let raw = t.opt_f64("quantity").unwrap();
            let norm = t.opt_f64("quantity_normalised").unwrap();
            let shift = t.f64("normalisation_log2_shift").unwrap();
            assert_eq!(t.str("peptidoform").unwrap()[N], "NULLPEP");
            assert_eq!(norm[N], None);
            assert_eq!(norm[N + 1], Some(0.0));
            // Raw run 5 is 1 log2 above the other five; centred, its shift is +5/6 and
            // theirs -1/6.
            let want = if r == 5 { 5.0 / 6.0 } else { -1.0 / 6.0 };
            for i in [0, N / 2, N - 1] {
                assert!(
                    (shift[i] - want).abs() < 0.03,
                    "run {r} row {i}: {}",
                    shift[i]
                );
                let back = (raw[i].unwrap() / norm[i].unwrap()).log2();
                assert!((back - shift[i]).abs() < 1e-12);
            }
            assert!(std::path::Path::new(&format!("{out}.report.json")).exists());
        }
        let s = TableFile::open(&shifts_out).unwrap();
        assert_eq!(s.nrows, w.n_shift_rows as usize);
        assert_eq!(s.i32("bin").unwrap()[0], -1);
        assert!(s.has_column("log2_median"));
        // Byte-identical on a second run.
        let again: Vec<String> = (0..6)
            .map(|r| test_path(&format!("pqn_again{r}.parquet")))
            .collect();
        let shifts_again = test_path("shifts_again.parquet");
        let w2 = run(NormaliseParams {
            inputs: &inputs,
            outputs: &again,
            shifts_out: &shifts_again,
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        for (a, b) in w.tables.iter().zip(&w2.tables) {
            assert_eq!(a.content_hash, b.content_hash);
        }
        assert_eq!(w.shifts.content_hash, w2.shifts.content_hash);
    }

    /// `stable_set` on a synthetic six-run, three-population design reproduces the
    /// prototype it was confirmed with (`bench/quant_methods/synth_ref.py`, case 2), with
    /// the automatic subspace dimension (1 here) and with a fixed one of 2.
    #[test]
    fn stable_set_reproduces_the_prototype() {
        const PICK: [(usize, u32); 41] = [
            (0, 0),
            (0, 600),
            (0, 1194),
            (0, 1791),
            (0, 2395),
            (0, 3002),
            (0, 3594),
            (1, 192),
            (1, 808),
            (1, 1398),
            (1, 2013),
            (1, 2609),
            (1, 3207),
            (1, 3815),
            (2, 425),
            (2, 1024),
            (2, 1623),
            (2, 2244),
            (2, 2841),
            (2, 3456),
            (3, 42),
            (3, 642),
            (3, 1234),
            (3, 1835),
            (3, 2447),
            (3, 3057),
            (3, 3660),
            (4, 250),
            (4, 842),
            (4, 1431),
            (4, 2020),
            (4, 2613),
            (4, 3215),
            (4, 3802),
            (5, 421),
            (5, 1010),
            (5, 1619),
            (5, 2208),
            (5, 2820),
            (5, 3419),
            (5, 3998),
        ];
        const SHIFT_AUTO: [f64; 41] = [
            0.2081193814881711,
            0.16154291806605517,
            0.10822892817927426,
            0.06978295445234442,
            0.1526196442998296,
            0.06978295445234442,
            0.11590257881605202,
            -0.05450668041266193,
            -0.05501590700882091,
            -0.057900460972608606,
            -0.056675466867475176,
            -0.057110680279670124,
            -0.055750916211164694,
            -0.06314971822873224,
            -0.028302939471257556,
            -0.17595963404932521,
            -0.025956672872953404,
            0.04509055769599732,
            -0.11032369944583176,
            0.00021372234831014755,
            0.37203524989201325,
            0.37203524989201325,
            0.3689585217430401,
            0.35208463038107635,
            0.3518267935980037,
            0.3679898582446516,
            0.35743482851559144,
            -0.21274453952712366,
            -0.31447857241869054,
            -0.24085709333732616,
            -0.1983860252770792,
            -0.19191475615757017,
            -0.2062688433959168,
            -0.2130039594945824,
            -0.0241911220850819,
            -0.27848594488126865,
            -0.23850735996098463,
            -0.05164060812486951,
            -0.04082373218476789,
            -0.1334552683211189,
            -0.024191122085082062,
        ];
        const SHIFT_K2: [f64; 41] = [
            0.18283599618533405,
            0.16091374880794654,
            0.12480561916121326,
            0.11267122675909746,
            0.15496521353012058,
            0.11267122675909746,
            0.1304884614937568,
            -0.05152137811368726,
            -0.06186312906131333,
            -0.04953595225075865,
            -0.049964624735430514,
            -0.0618631290613135,
            -0.0618631290613135,
            -0.04953595225075882,
            -0.038880659888248965,
            -0.14171977838777935,
            -0.03702941711198549,
            0.00416857009921584,
            -0.10598230959040505,
            -0.011069073911856065,
            0.3712166383281463,
            0.3712166383281463,
            0.3712166383281463,
            0.34724322418149534,
            0.34724322418149534,
            0.37121663832814655,
            0.3546615485285733,
            -0.22431781005714777,
            -0.2733108038190135,
            -0.24068802085949148,
            -0.22431781005714777,
            -0.22431781005714777,
            -0.22431781005714777,
            -0.22431781005714777,
            -0.05885311759689678,
            -0.21762811533718113,
            -0.21762811533718093,
            -0.05885311759689678,
            -0.05885311759689678,
            -0.13193390740052513,
            -0.058853117596896945,
        ];
        let off = [
            (0.3, 0.1),
            (0.0, 0.0),
            (-0.2, 0.2),
            (0.4, 0.4),
            (-0.3, -0.1),
            (0.1, -0.3),
        ];
        let mut g = Rng(17);
        // per run: (candidate, log2 quantity, apex)
        let mut rows: Vec<Vec<(u32, f64, f64)>> = vec![Vec::new(); 6];
        for i in 0..4000u32 {
            let sp = match i % 20 {
                0..=12 => 0,
                13..=16 => 1,
                _ => 2,
            };
            let ab = 15.0 + 6.0 * g.uniform();
            let rt = 3000.0 * g.uniform();
            let sd = 0.05 + 0.3 * (21.0 - ab) / 6.0;
            for (r, &(a, b)) in off.iter().enumerate() {
                let fold = match (sp, r < 3) {
                    (1, true) => 0.5,
                    (1, false) => -0.5,
                    (2, true) => -1.0,
                    (2, false) => 1.0,
                    _ => 0.0,
                };
                let o = a + (b - a) * rt / 3000.0;
                let y = ab + fold + o + sd * g.normal();
                let rr = rt + 2.0 * g.normal();
                let miss = g.uniform() < 0.25 * (21.0 - ab) / 6.0;
                if !miss {
                    rows[r].push((i, y, rr));
                }
            }
        }
        assert_eq!(rows.iter().map(|r| r.len()).sum::<usize>(), 21082);
        let inputs: Vec<String> = (0..6)
            .map(|r| test_path(&format!("ss_pq{r}.parquet")))
            .collect();
        for (r, path) in inputs.iter().enumerate() {
            let n = rows[r].len();
            mumdia_io::table::write_table(
                path,
                vec![
                    Col::U32("candidate_id".into(), rows[r].iter().map(|x| x.0).collect()),
                    Col::Str(
                        "peptidoform".into(),
                        rows[r].iter().map(|x| format!("PEP{}", x.0)).collect(),
                    ),
                    Col::I32("charge".into(), vec![2; n]),
                    Col::Str("protein_group".into(), vec!["PG".into(); n]),
                    Col::OptF64(
                        "quantity".into(),
                        rows[r].iter().map(|x| Some(x.1.exp2())).collect(),
                    ),
                    Col::Str("quant_status".into(), vec!["quantified".into(); n]),
                    Col::I32("n_fragments_used".into(), vec![3; n]),
                    Col::OptF64(
                        "integration_apex_rt".into(),
                        rows[r].iter().map(|x| Some(x.2)).collect(),
                    ),
                ],
            )
            .unwrap();
        }
        for (rank, want) in [
            (None, &SHIFT_AUTO),
            (Some(1), &SHIFT_AUTO),
            (Some(2), &SHIFT_K2),
        ] {
            let outputs: Vec<String> = (0..6)
                .map(|r| test_path(&format!("ss_out{r}.parquet")))
                .collect();
            let cfg = QuantConfig {
                normalisation: CrossRunNormalisation::StableSet,
                normalisation_rank: rank,
                ..QuantConfig::default()
            };
            let shifts_out = test_path("ss_shifts.parquet");
            run(NormaliseParams {
                inputs: &inputs,
                outputs: &outputs,
                shifts_out: &shifts_out,
                cfg: &cfg,
                config_hash: "test",
            })
            .unwrap();
            let shift: Vec<Vec<f64>> = outputs
                .iter()
                .map(|o| {
                    TableFile::open(o)
                        .unwrap()
                        .f64("normalisation_log2_shift")
                        .unwrap()
                })
                .collect();
            for (&(r, c), &w) in PICK.iter().zip(want.iter()) {
                let k = rows[r].iter().position(|x| x.0 == c).unwrap();
                let got = shift[r][k];
                assert!(
                    (got - w).abs() < 1e-9,
                    "rank {rank:?}, run {r}, ion {c}: {got} against {w}"
                );
            }
            // The normalised quantity is quantity / 2^shift.
            let t = TableFile::open(&outputs[0]).unwrap();
            let (q, qn) = (
                t.opt_f64("quantity").unwrap(),
                t.opt_f64("quantity_normalised").unwrap(),
            );
            assert_eq!(qn[0], Some(q[0].unwrap() / shift[0][0].exp2()));
        }
        // A fixed rank that leaves no residual dimension is refused.
        let cfg = QuantConfig {
            normalisation: CrossRunNormalisation::StableSet,
            normalisation_rank: Some(5),
            ..QuantConfig::default()
        };
        let outputs: Vec<String> = (0..6)
            .map(|r| test_path(&format!("ss_bad{r}.parquet")))
            .collect();
        assert!(run(NormaliseParams {
            inputs: &inputs,
            outputs: &outputs,
            shifts_out: &test_path("ss_bad_shifts.parquet"),
            cfg: &cfg,
            config_hash: "test",
        })
        .is_err());
    }
}
