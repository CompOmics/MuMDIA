//! Sidecar clients over the file contract (docs/13_sidecars.md): write an input
//! Parquet, invoke the Python worker as a subprocess, read the output Parquet.
//! The contract is the files and their schema, so a sidecar can be replaced by a
//! native implementation without changing callers.

use std::collections::HashMap;
use std::process::Command;

use anyhow::{bail, Context, Result};
use mumdia_io::table::{write_table, Col, TableFile};
use tracing::{info, warn};

/// Per candidate row: `(ion byte, ordinal, fragment charge)` -> linear predicted intensity.
/// The charge is 1 for every series a single-charge MS2PIP model emits and 2 for the
/// `b2`/`y2` series of the `*ch2` models.
pub type FragmentIntensityMap = HashMap<u32, HashMap<(u8, u16, u8), f32>>;

/// Resolve a sidecar worker script path so a deployed binary finds its workers
/// regardless of the working directory: try the configured dir relative to the
/// the binary's own directory, then `<exe_dir>/scripts`, and the current working
/// directory LAST. Falls back to the directory-relative path (so the eventual error
/// names it) if none of those exist.
///
/// The working directory used to be tried first. `python::resolve_script_dir` was
/// reordered away from exactly that and documents why at length: the shipped default is
/// the relative `"scripts"`, which both sidecar example configs carry, so unpacking a
/// dataset archive, `cd`-ing into it and running with an example config executed any
/// worker the archive happened to contain. That resolver only claims a directory holding
/// `mbr_worker.py` or `deeplc_worker.py`, so a directory with any of the other ten
/// workers reached this function still relative, and this function ran it (docs/31 F3).
/// An absolute directory is taken as given: naming one is how a user is unambiguous.
pub fn resolve_script(dir: &str, worker: &str) -> String {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.to_path_buf()));
    resolve_script_in(dir, worker, exe_dir.as_deref())
}

/// [`resolve_script`] with the executable's directory supplied, so the ordering can be
/// tested without a second binary.
fn resolve_script_in(dir: &str, worker: &str, exe_dir: Option<&std::path::Path>) -> String {
    let dir_rel = format!("{dir}/{worker}");
    if std::path::Path::new(dir).is_absolute() {
        return dir_rel;
    }
    if let Some(base) = exe_dir {
        for cand in [
            base.join(dir).join(worker),
            base.join("scripts").join(worker),
        ] {
            if cand.exists() {
                return cand.to_string_lossy().into_owned();
            }
        }
    }
    dir_rel
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
    // Per-invocation names. Fixed ones made two concurrent runs clobber each other, and
    // silently rather than loudly: the readback key is a row index into each process's own
    // request, so two runs over tables of the SAME row count -- which is the most likely
    // reason to run two at once -- swapped each other's results instead of erroring.
    // `align_sidecar_scores` catches a coverage mismatch, not an equal-length swap. The
    // PIN/NN path was already PID-qualified for exactly this reason; these three were not.
    let pid = std::process::id();
    let inp = format!("{workdir}/ms2pip_in_{pid}.parquet");
    let outp = format!("{workdir}/ms2pip_out_{pid}.parquet");
    write_table(
        &inp,
        vec![
            Col::U32("id".into(), ids.to_vec()),
            Col::Str("peptidoform".into(), peptidoforms.to_vec()),
            Col::I32("charge".into(), charges.to_vec()),
        ],
    )?;
    // The worker sizes its MS2PIP process pool from this. Left to itself it capped the
    // pool at min(8, cpu_count), which on the 9.8M-peptidoform HYE library left 24 of
    // the 32 requested cores idle for the whole prediction.
    let processes = rayon::current_num_threads().max(1).to_string();
    info!(n = ids.len(), model, processes = %processes, "sidecar: running MS2PIP");
    run_worker(python, script, &[&inp, &outp, model, &processes], false)
        .context("MS2PIP worker failed")?;

    let t = TableFile::open(&outp)?;
    let oid = t.u32("id")?;
    let ion = t.str("ion_type")?;
    let ord = t.i32("ordinal")?;
    let inten = t.f32("intensity")?;
    // Older workers wrote no `frag_charge`: every row was a singly charged fragment.
    let fch = if t.has_column("frag_charge") {
        Some(t.i32("frag_charge")?)
    } else {
        None
    };
    // Returned ids must be requested ones and each (ion, ordinal, charge) may appear once
    // per id; coverage of the requested set is the caller's decision (docs/29 #17).
    let requested: std::collections::HashSet<u32> = ids.iter().copied().collect();
    let mut map: FragmentIntensityMap = HashMap::new();
    for i in 0..t.nrows {
        if !requested.contains(&oid[i]) {
            bail!(
                "MS2PIP worker returned id {}, which was not among the {} peptidoforms \
                 requested",
                oid[i],
                ids.len()
            );
        }
        let ib = ion[i].as_bytes().first().copied().unwrap_or(b'?');
        let z = fch.as_ref().map(|c| c[i].clamp(1, 255) as u8).unwrap_or(1);
        if map
            .entry(oid[i])
            .or_default()
            .insert((ib, ord[i] as u16, z), inten[i])
            .is_some()
        {
            bail!(
                "MS2PIP worker returned fragment {}{} charge {z} of id {} more than once",
                ion[i],
                ord[i],
                oid[i]
            );
        }
    }
    Ok(map)
}

/// Installed version of a Python distribution as the interpreter reports it
/// (`crate::python::module_version`, read from package metadata so the probe does not
/// import the package). None when the interpreter cannot be run or the distribution is
/// not installed.
pub fn module_version(python: &str, module: &str) -> Option<String> {
    crate::python::module_version(std::path::Path::new(python), module)
}

/// Refuse a DeepLC interpreter older than `MIN_DEEPLC_VERSION`. The default
/// retention-time workflow calibrates base-model predictions per run without a
/// fine-tune, and that rests on the base model not memorising its anchors, which
/// 4.0.0a2 did (docs/08 section 4b). Both worker scripts repeat the check, but
/// failing here is cheaper than failing after the input table has been written.
pub fn require_deeplc_version(python: &str) -> Result<String> {
    use mumdia_core::constants::{parse_version3, MIN_DEEPLC_VERSION};
    let (ma, mi, pa) = MIN_DEEPLC_VERSION;
    let v = module_version(python, "deeplc").ok_or_else(|| {
        anyhow::anyhow!(
            "cannot determine the deeplc version through {python} (is DeepLC >= {ma}.{mi}.{pa} installed there?)"
        )
    })?;
    match parse_version3(&v) {
        Some(t) if t >= MIN_DEEPLC_VERSION => Ok(v),
        _ => bail!(
            "deeplc {v} at {python} is older than the required {ma}.{mi}.{pa};              upgrade with `pip install 'deeplc>={ma}.{mi}.{pa}'`"
        ),
    }
}

/// `deeplc-<version>-<suffix>` when the interpreter answers, `deeplc-<suffix>` when there
/// is none to ask: the manifest's RT identity should say which DeepLC release produced the
/// library, not only the recipe (docs/30, model identity).
pub fn deeplc_identity(python: Option<&str>, suffix: &str) -> String {
    match python.and_then(|py| module_version(py, "deeplc")) {
        Some(v) => format!("deeplc-{v}-{suffix}"),
        None => format!("deeplc-{suffix}"),
    }
}

/// Read the `<lib_out>.summary.json` the fine-tune worker writes beside a rewritten
/// library and warn when rows kept their imported iRT, so a mixed RT source is visible
/// in the log rather than only in the file (docs/30 R6).
fn warn_on_retained_imported(lib_out: &str) {
    let path = format!("{lib_out}.summary.json");
    let Ok(v) = mumdia_io::json::read_json::<serde_json::Value>(&path) else {
        return;
    };
    let n = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
    let retained = n("retained_imported");
    if retained > 0 {
        warn!(
            rows = n("rows"),
            repredicted = n("repredicted"),
            retained_imported = retained,
            retained_non_standard = n("retained_non_standard"),
            retained_no_prediction = n("retained_no_prediction"),
            summary = %path,
            "sidecar: the rewritten library keeps the imported iRT on some rows, so its RT \
             source is mixed; see the summary for the counts"
        );
    }
}

/// DeepLC: predict retention time per peptidoform. Returns `id -> predicted_rt`.
pub fn run_deeplc(
    python: &str,
    script: &str,
    workdir: &str,
    ids: &[u32],
    peptidoforms: &[String],
) -> Result<HashMap<u32, f32>> {
    require_deeplc_version(python)?;
    std::fs::create_dir_all(workdir).ok();
    // Per-invocation names; see the note in the MS2PIP helper above.
    let pid = std::process::id();
    let inp = format!("{workdir}/deeplc_in_{pid}.parquet");
    let outp = format!("{workdir}/deeplc_out_{pid}.parquet");
    write_table(
        &inp,
        vec![
            Col::U32("id".into(), ids.to_vec()),
            Col::Str("peptidoform".into(), peptidoforms.to_vec()),
        ],
    )?;
    info!(n = ids.len(), "sidecar: running DeepLC");
    run_worker(python, script, &[&inp, &outp], true).context("DeepLC worker failed")?;

    let t = TableFile::open(&outp)?;
    let oid = t.u32("id")?;
    let rt = t.f32("predicted_rt")?;
    // Returned ids must be a subset of the requested ones, each at most once. A repeated
    // id used to overwrite silently and an unrequested one was kept; coverage (ids with
    // no prediction) is the caller's to decide, and it drops those candidates rather
    // than substituting a value (docs/29 #17).
    let requested: std::collections::HashSet<u32> = ids.iter().copied().collect();
    let mut map: HashMap<u32, f32> = HashMap::with_capacity(oid.len());
    for (id, value) in oid.into_iter().zip(rt) {
        if !requested.contains(&id) {
            bail!(
                "DeepLC worker returned id {id}, which was not among the {} peptidoforms \
                 requested",
                ids.len()
            );
        }
        if map.insert(id, value).is_some() {
            bail!("DeepLC worker returned id {id} more than once");
        }
    }
    Ok(map)
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
    rng_seed: u64,
) -> Result<()> {
    require_deeplc_version(python)?;
    info!(
        lib_in,
        seed,
        lib_out,
        epochs,
        patience,
        q_train,
        batch,
        window_holdout_frac,
        rng_seed,
        "sidecar: running DeepLC multitask fine-tune"
    );
    let ep = epochs.to_string();
    let pa = patience.to_string();
    let qt = q_train.to_string();
    let ba = batch.to_string();
    let hf = window_holdout_frac.to_string();
    // Seed the fine-tune. Unseeded, the draw varies enough to change results: the
    // held-out RT window p95 moved 150-211 s across two draws of one benchmark arm,
    // worth about 2 percent of peptides, which made single-run comparisons of
    // window sizing or library variants unreadable. Kernel-level nondeterminism
    // remains, so this narrows the variance rather than removing it
    // (docs/14_build_test_deploy_gotchas.md).
    let rs = rng_seed.to_string();
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
            "--seed",
            &rs,
        ],
        true,
    )
    .context("DeepLC fine-tune failed")?;
    warn_on_retained_imported(lib_out);
    Ok(())
}

/// Multi-head calibration of the library's `predicted_irt` against this run's anchors.
///
/// The fine-tune worker in its third mode: same positional contract, same reference
/// (the confident seed PSMs), but instead of transfer-learning it fits
/// `MultiHeadRidgeCalibration` over the base model's best-correlating LC-setup heads and
/// writes the calibrated retention times into the library. See
/// `RtImTrainConfig::multihead_calibration` for why one head plus a monotone curve is not
/// the same thing. Positional contract:
/// `deeplc_finetune.py <lib_in> <seed> <lib_out> --multihead <n_heads>`.
#[allow(clippy::too_many_arguments)]
pub fn run_deeplc_multihead(
    python: &str,
    script: &str,
    lib_in: &str,
    seed: &str,
    lib_out: &str,
    n_heads: usize,
    q_train: f64,
    window_holdout_frac: f64,
    threads: usize,
) -> Result<()> {
    require_deeplc_version(python)?;
    info!(
        lib_in,
        seed, lib_out, n_heads, q_train, "sidecar: calibrating DeepLC over multiple heads"
    );
    let nh = n_heads.to_string();
    let qt = q_train.to_string();
    let hf = window_holdout_frac.to_string();
    let th = threads.max(1).to_string();
    run_worker(
        python,
        script,
        &[
            lib_in,
            seed,
            lib_out,
            "--multihead",
            &nh,
            "--q-train",
            &qt,
            "--window-holdout-frac",
            &hf,
            "--threads",
            &th,
            "--predict-threads",
            &th,
        ],
        true,
    )
    .context("DeepLC multi-head calibration failed")?;
    warn_on_retained_imported(lib_out);
    Ok(())
}

/// DeepLC base-model re-prediction of an imported library's `predicted_irt`: the
/// fine-tune worker with `--no-finetune`, so the table rewrite (targets predicted on their
/// peptidoform, decoys on the DECOY_-stripped sequence, rows with non-standard residues
/// keeping the imported value) is the one the fine-tune path uses. Positional contract:
/// `deeplc_finetune.py <lib_in> - <lib_out> --no-finetune`. Prediction is forward-only,
/// so it takes the engine's full thread count rather than the fine-tune's bounded pool.
pub fn run_deeplc_repredict(
    python: &str,
    script: &str,
    lib_in: &str,
    lib_out: &str,
    threads: usize,
) -> Result<()> {
    require_deeplc_version(python)?;
    info!(
        lib_in,
        lib_out, threads, "sidecar: re-predicting the library iRT with the DeepLC base model"
    );
    let th = threads.max(1).to_string();
    run_worker(
        python,
        script,
        &[
            lib_in,
            "-",
            lib_out,
            "--no-finetune",
            "--threads",
            &th,
            "--predict-threads",
            &th,
        ],
        true,
    )
    .context("DeepLC library re-prediction failed")?;
    warn_on_retained_imported(lib_out);
    Ok(())
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
    // Output is INHERITED, not captured, on purpose: the NN rescorer prints
    // per-iteration progress across hours and a user watching a run needs to see
    // it live. The cost is that on failure the worker's traceback has already
    // scrolled past, so the error has to say enough to act on: which interpreter,
    // which script, and with what arguments. It previously said only
    // "exited with status", which named neither the environment nor the call.
    let status = cmd
        .status()
        .with_context(|| format!("spawning {python} {script}"))?;
    if !status.success() {
        let argv = args
            .iter()
            .map(|a| {
                if a.contains(' ') {
                    format!("\"{a}\"")
                } else {
                    (*a).to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        bail!(
            "sidecar worker failed: {script} exited with {status}.\n\
             interpreter: {python}\n\
             command: {python} {script} {argv}\n\
             The worker's own output, including any Python traceback, is above this \
             error. Reproduce it by running that command directly. \
             `mumdia doctor --config <config>` checks the environment."
        );
    }
    Ok(())
}

#[cfg(test)]
mod resolve_tests {
    use super::resolve_script_in;
    use std::path::Path;

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_resolve_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_shipped_directory_beside_the_binary_wins_over_the_working_directory() {
        // docs/31 F3. Both locations hold a worker of the same name; the one that ships
        // with the binary must win, because the working directory can be an untrusted
        // dataset the user merely unpacked and `cd`-ed into.
        let exe = scratch("exe");
        std::fs::create_dir_all(exe.join("scripts")).unwrap();
        std::fs::write(exe.join("scripts").join("ms2pip_worker.py"), b"# shipped").unwrap();
        let got = resolve_script_in("scripts", "ms2pip_worker.py", Some(&exe));
        assert_eq!(
            got,
            exe.join("scripts")
                .join("ms2pip_worker.py")
                .to_string_lossy()
        );
        assert_ne!(got, "scripts/ms2pip_worker.py");
        let _ = std::fs::remove_dir_all(&exe);
    }

    #[test]
    fn an_absolute_directory_is_taken_as_given() {
        let abs = if cfg!(windows) {
            "C:/opt/mumdia/scripts"
        } else {
            "/opt/mumdia/scripts"
        };
        let exe = scratch("abs");
        std::fs::create_dir_all(exe.join("scripts")).unwrap();
        std::fs::write(exe.join("scripts").join("mbr_worker.py"), b"# shipped").unwrap();
        assert_eq!(
            resolve_script_in(abs, "mbr_worker.py", Some(&exe)),
            format!("{abs}/mbr_worker.py"),
            "naming a directory outright is how a user is unambiguous"
        );
        let _ = std::fs::remove_dir_all(&exe);
    }

    #[test]
    fn nothing_beside_the_binary_falls_back_to_the_relative_path_for_the_error() {
        let exe = scratch("empty");
        assert_eq!(
            resolve_script_in("scripts", "deeplc_worker.py", Some(&exe)),
            "scripts/deeplc_worker.py"
        );
        assert_eq!(
            resolve_script_in("scripts", "deeplc_worker.py", None::<&Path>),
            "scripts/deeplc_worker.py"
        );
        let _ = std::fs::remove_dir_all(&exe);
    }
}
