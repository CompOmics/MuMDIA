//! Sidecar clients over the file contract (docs/13_sidecars.md): write an input
//! Parquet, invoke the Python worker as a subprocess, read the output Parquet.
//! The contract is the files and their schema, so a sidecar can be replaced by a
//! native implementation without changing callers.

use std::collections::HashMap;
use std::process::Command;

use anyhow::{bail, Context, Result};
use mumdia_io::table::{write_table, Col, Table};
use tracing::info;

type FragmentIntensityMap = HashMap<u32, HashMap<(u8, u16), f32>>;

/// Resolve a sidecar worker script path so a deployed binary finds its workers
/// regardless of the working directory: try the configured dir relative to the
/// CWD, then relative to the binary's own directory, then `<exe_dir>/scripts`.
/// Falls back to the CWD-relative path (so the eventual error names it) if none
/// of those exist.
pub fn resolve_script(dir: &str, worker: &str) -> String {
    let cwd_rel = format!("{dir}/{worker}");
    if std::path::Path::new(&cwd_rel).exists() {
        return cwd_rel;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(base) = exe.parent() {
            for cand in [
                base.join(dir).join(worker),
                base.join("scripts").join(worker),
            ] {
                if cand.exists() {
                    return cand.to_string_lossy().into_owned();
                }
            }
        }
    }
    cwd_rel
}

/// MS2PIP: predict singly-charged b/y intensities per (peptidoform, charge).
/// Returns `candidate_id -> (ion_byte b'b'/b'y', ordinal) -> linear intensity`.
pub fn run_ms2pip(
    python: &str,
    script: &str,
    workdir: &str,
    ids: &[u32],
    peptidoforms: &[String],
    charges: &[i32],
    model: &str,
) -> Result<FragmentIntensityMap> {
    std::fs::create_dir_all(workdir).ok();
    let inp = format!("{workdir}/ms2pip_in.parquet");
    let outp = format!("{workdir}/ms2pip_out.parquet");
    write_table(
        &inp,
        vec![
            Col::U32("id".into(), ids.to_vec()),
            Col::Str("peptidoform".into(), peptidoforms.to_vec()),
            Col::I32("charge".into(), charges.to_vec()),
        ],
    )?;
    info!(n = ids.len(), model, "sidecar: running MS2PIP");
    run_worker(python, script, &[&inp, &outp, model], false).context("MS2PIP worker failed")?;

    let t = Table::read(&outp)?;
    let oid = t.u32("id")?;
    let ion = t.str("ion_type")?;
    let ord = t.i32("ordinal")?;
    let inten = t.f32("intensity")?;
    let mut map: HashMap<u32, HashMap<(u8, u16), f32>> = HashMap::new();
    for i in 0..t.nrows {
        let ib = ion[i].as_bytes().first().copied().unwrap_or(b'?');
        map.entry(oid[i])
            .or_default()
            .insert((ib, ord[i] as u16), inten[i]);
    }
    Ok(map)
}

/// DeepLC: predict retention time per peptidoform. Returns `id -> predicted_rt`.
pub fn run_deeplc(
    python: &str,
    script: &str,
    workdir: &str,
    ids: &[u32],
    peptidoforms: &[String],
) -> Result<HashMap<u32, f32>> {
    std::fs::create_dir_all(workdir).ok();
    let inp = format!("{workdir}/deeplc_in.parquet");
    let outp = format!("{workdir}/deeplc_out.parquet");
    write_table(
        &inp,
        vec![
            Col::U32("id".into(), ids.to_vec()),
            Col::Str("peptidoform".into(), peptidoforms.to_vec()),
        ],
    )?;
    info!(n = ids.len(), "sidecar: running DeepLC");
    run_worker(python, script, &[&inp, &outp], true).context("DeepLC worker failed")?;

    let t = Table::read(&outp)?;
    let oid = t.u32("id")?;
    let rt = t.f32("predicted_rt")?;
    Ok(oid.into_iter().zip(rt).collect())
}

/// DeepLC multitask fine-tune: adapt the RT model to this run's confident seed
/// PSMs and rewrite the supplied library's `predicted_irt`. Positional contract:
/// `deeplc_finetune.py <lib_in> <seed> <lib_out>`.
#[allow(clippy::too_many_arguments)]
pub fn run_deeplc_finetune(
    python: &str,
    script: &str,
    lib_in: &str,
    seed: &str,
    lib_out: &str,
    epochs: usize,
    patience: usize,
    q_train: f64,
    batch: usize,
    window_holdout_frac: f64,
) -> Result<()> {
    info!(
        lib_in,
        seed,
        lib_out,
        epochs,
        patience,
        q_train,
        batch,
        window_holdout_frac,
        "sidecar: running DeepLC multitask fine-tune"
    );
    let ep = epochs.to_string();
    let pa = patience.to_string();
    let qt = q_train.to_string();
    let ba = batch.to_string();
    let hf = window_holdout_frac.to_string();
    run_worker(
        python,
        script,
        &[
            lib_in,
            seed,
            lib_out,
            "--epochs",
            &ep,
            "--patience",
            &pa,
            "--q-train",
            &qt,
            "--batch",
            &ba,
            "--window-holdout-frac",
            &hf,
        ],
        true,
    )
    .context("DeepLC fine-tune failed")
}

/// MBR transfer (Stage D3): match-between-runs identification transfer over the
/// experiment-wide scored table + per-run psms. Positional contract:
/// `mbr_worker.py <scored_combined> <psms_csv> <out_transferred> [flags]`, where
/// `psms_csv` is the per-run psms paths joined by ',' in `source` order.
#[allow(clippy::too_many_arguments)]
pub fn run_mbr(
    python: &str,
    script: &str,
    scored: &str,
    psms: &[String],
    out: &str,
    out_scored: Option<&str>,
    frag: &[String],
    q_anchor: f64,
    min_anchor_runs: usize,
    q_transfer: f64,
    consensus_corr_min: f64,
    seed: u64,
) -> Result<()> {
    let psms_csv = psms.join(",");
    info!(
        scored,
        out,
        runs = psms.len(),
        q_anchor,
        min_anchor_runs,
        q_transfer,
        consensus_corr_min,
        "sidecar: running MBR transfer"
    );
    let qa = q_anchor.to_string();
    let mar = min_anchor_runs.to_string();
    let qt = q_transfer.to_string();
    let sd = seed.to_string();
    let cm = consensus_corr_min.to_string();
    let frag_csv = frag.join(",");
    let mut args: Vec<&str> = vec![
        scored,
        &psms_csv,
        out,
        "--q-anchor",
        &qa,
        "--min-anchor-runs",
        &mar,
        "--q-transfer",
        &qt,
        "--seed",
        &sd,
    ];
    if let Some(os) = out_scored {
        args.extend_from_slice(&["--out-scored", os]);
    }
    if !frag.is_empty() && consensus_corr_min > 0.0 {
        args.extend_from_slice(&["--frag-csv", &frag_csv, "--consensus-corr-min", &cm]);
    }
    run_worker(python, script, &args, false).context("MBR transfer worker failed")
}

/// Invoke a Python worker: `python script arg...`. `utf8` forces UTF-8 I/O
/// (DeepLC/Keras crash on the Windows cp1252 console otherwise).
fn run_worker(python: &str, script: &str, args: &[&str], utf8: bool) -> Result<()> {
    let mut cmd = Command::new(python);
    cmd.arg(script);
    for a in args {
        cmd.arg(a);
    }
    if utf8 {
        cmd.env("PYTHONUTF8", "1").env("PYTHONIOENCODING", "utf-8");
    }
    let status = cmd
        .status()
        .with_context(|| format!("spawning {python} {script}"))?;
    if !status.success() {
        bail!("worker {script} exited with {status}");
    }
    Ok(())
}
