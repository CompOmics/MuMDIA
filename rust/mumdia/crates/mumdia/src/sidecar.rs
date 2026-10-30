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
    let (inp, outp) = fragment_request(workdir, "ms2pip", ids, peptidoforms, charges, None)?;
    // The worker sizes its MS2PIP process pool from this. Left to itself it capped the
    // pool at min(8, cpu_count), which on the 9.8M-peptidoform HYE library left 24 of
    // the 32 requested cores idle for the whole prediction.
    let processes = rayon::current_num_threads().max(1).to_string();
    info!(n = ids.len(), model, processes = %processes, "sidecar: running MS2PIP");
    run_worker(python, script, &[&inp, &outp, model, &processes], false)
        .context("MS2PIP worker failed")?;

    read_fragment_intensities(&outp, ids, "MS2PIP")
}

/// Write the `id, peptidoform, charge` request the fragment-intensity workers and the
/// IM2Deep worker read, and name the output file. Returns `(input, output)`. `extra` appends
/// one Float64 column (`(name, values)`, one value per id): IM2Deep takes the precursor m/z
/// to convert its CCS to 1/K0. Without it the file is the three-column request.
///
/// Per-invocation names. Fixed ones made two concurrent runs clobber each other, and
/// silently rather than loudly: the readback key is a row index into each process's own
/// request, so two runs over tables of the SAME row count -- which is the most likely
/// reason to run two at once -- swapped each other's results instead of erroring.
/// `align_sidecar_scores` catches a coverage mismatch, not an equal-length swap. The
/// PIN/NN path was already PID-qualified for exactly this reason; these were not.
fn fragment_request(
    workdir: &str,
    tag: &str,
    ids: &[u32],
    peptidoforms: &[String],
    charges: &[i32],
    extra: Option<(&str, &[f64])>,
) -> Result<(String, String)> {
    std::fs::create_dir_all(workdir).ok();
    let pid = std::process::id();
    let inp = format!("{workdir}/{tag}_in_{pid}.parquet");
    let outp = format!("{workdir}/{tag}_out_{pid}.parquet");
    let mut cols = vec![
        Col::U32("id".into(), ids.to_vec()),
        Col::Str("peptidoform".into(), peptidoforms.to_vec()),
        Col::I32("charge".into(), charges.to_vec()),
    ];
    if let Some((name, values)) = extra {
        cols.push(Col::F64(name.into(), values.to_vec()));
    }
    write_table(&inp, cols)?;
    Ok((inp, outp))
}

/// Read a worker's `id -> value` table: one scalar per requested id, such as DeepLC's
/// `predicted_rt` or IM2Deep's `predicted_im`.
///
/// Shared for the same reason as [`read_fragment_intensities`]. Returned ids must be
/// requested ones, each at most once: a repeated id used to overwrite silently and an
/// unrequested one was kept. Coverage, the ids with no row, is the caller's to decide, and
/// predict-frag drops those candidates with their pairs rather than substituting a value
/// (docs/29 #17). `finite_positive` also refuses a value that is not finite and above zero,
/// for a quantity that cannot be anything else (1/K0); a retention time can be negative on
/// a model's scale, so DeepLC's readback leaves it off. `who` names the worker and `noun`
/// what was requested, in the errors.
fn read_id_scalars<T: Copy + Into<f64> + std::fmt::Display>(
    out_path: &str,
    ids: &[u32],
    who: &str,
    noun: &str,
    column: &str,
    read: fn(&TableFile, &str) -> Result<Vec<T>>,
    finite_positive: bool,
) -> Result<HashMap<u32, T>> {
    let t = TableFile::open(out_path)?;
    let oid = t.u32("id")?;
    let values = read(&t, column)?;
    let requested: std::collections::HashSet<u32> = ids.iter().copied().collect();
    let mut map: HashMap<u32, T> = HashMap::with_capacity(oid.len());
    for (id, value) in oid.into_iter().zip(values) {
        if !requested.contains(&id) {
            bail!(
                "{who} worker returned id {id}, which was not among the {} {noun} requested",
                ids.len()
            );
        }
        if finite_positive {
            let v: f64 = value.into();
            if !(v.is_finite() && v > 0.0) {
                bail!("{who} worker returned a {column} of {value} for id {id}; it must be finite and positive");
            }
        }
        if map.insert(id, value).is_some() {
            bail!("{who} worker returned id {id} more than once");
        }
    }
    Ok(map)
}

/// Read the five-column fragment-intensity table both workers write.
///
/// Shared rather than copied, so a predictor added later cannot arrive with its own
/// slightly different idea of the contract. `who` names the worker in the errors.
fn read_fragment_intensities(
    out_path: &str,
    ids: &[u32],
    who: &str,
) -> Result<FragmentIntensityMap> {
    let t = TableFile::open(out_path)?;
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
                "{who} worker returned id {}, which was not among the {} peptidoforms \
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
                "{who} worker returned fragment {}{} charge {z} of id {} more than once",
                ion[i],
                ord[i],
                oid[i]
            );
        }
    }
    Ok(map)
}

/// AlphaPeptDeep fragment intensities: the same contract as [`run_ms2pip`], a different
/// model behind it.
///
/// Positional contract: `peptdeep_worker.py <in> <out> <model> <nce> <instrument>
/// <processes>`. The device is chosen by `MUMDIA_PEPTDEEP_DEVICE` (auto|cuda|cpu) in the
/// worker, matching `MUMDIA_NN_DEVICE` in the rescorer rather than adding a second
/// convention; the worker prints the device it actually got, because on a whole-proteome
/// library the difference is an order of magnitude and a CPU-only torch looks the same
/// from outside.
#[allow(clippy::too_many_arguments)]
pub fn run_peptdeep(
    python: &str,
    script: &str,
    workdir: &str,
    ids: &[u32],
    peptidoforms: &[String],
    charges: &[i32],
    model: &str,
    nce: f64,
    instrument: &str,
) -> Result<FragmentIntensityMap> {
    let (inp, outp) = fragment_request(workdir, "peptdeep", ids, peptidoforms, charges, None)?;
    let nce_s = nce.to_string();
    let processes = rayon::current_num_threads().max(1).to_string();
    info!(
        n = ids.len(),
        model,
        nce,
        instrument,
        processes = %processes,
        "sidecar: running AlphaPeptDeep"
    );
    run_worker(
        python,
        script,
        &[&inp, &outp, model, &nce_s, instrument, &processes],
        false,
    )
    .context("AlphaPeptDeep worker failed")?;
    read_fragment_intensities(&outp, ids, "AlphaPeptDeep")
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
    require_version(python, "deeplc", mumdia_core::constants::MIN_DEEPLC_VERSION)
}

/// Refuse an interpreter whose `module` is older than `floor`, and return the version it
/// reports. The one check behind every sidecar version floor (`Role::version_floor`):
/// discovery and `doctor` judge with the same `python::judge_version`.
pub fn require_version(python: &str, module: &str, floor: (u32, u32, u32)) -> Result<String> {
    use crate::python::{fmt_version, judge_version, FloorCheck};
    let want = fmt_version(floor);
    match judge_version(module_version(python, module), floor) {
        FloorCheck::Met(v) => Ok(v),
        FloorCheck::TooOld(v) => bail!(
            "{module} {v} at {python} is older than the required {want}; \
             upgrade with `pip install '{module}>={want}'`"
        ),
        FloorCheck::Unknown => bail!(
            "cannot determine the {module} version through {python} \
             (is {module} >= {want} installed there?)"
        ),
    }
}

/// `deeplc-<version>-<suffix>` when the interpreter answers, `deeplc-<suffix>` when there
/// is none to ask: the manifest's RT identity should say which DeepLC release produced the
/// library, not only the recipe (docs/30, model identity).
pub fn deeplc_identity(python: Option<&str>, suffix: &str) -> String {
    model_identity(python, "deeplc", suffix)
}

/// `<module>-<version>-<suffix>` when the interpreter answers, `<module>-<suffix>` when there
/// is none to ask: the identity a library or a manifest records for a sidecar model, naming
/// the installed release as well as the recipe.
pub fn model_identity(python: Option<&str>, module: &str, suffix: &str) -> String {
    match python.and_then(|py| module_version(py, module)) {
        Some(v) => format!("{module}-{v}-{suffix}"),
        None => format!("{module}-{suffix}"),
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

/// Refuse a calibrated library that kept any row's input iRT, when that input was the
/// native placeholder a deferred FASTA build wrote (`predict_frag.defer_deeplc_to_multihead`):
/// such a row would keep an iRT from another model, on another scale, where the default
/// build would have had DeepLC's. Reads `<lib_out>.summary.json`.
pub fn require_every_row_repredicted(lib_out: &str) -> Result<()> {
    let path = format!("{lib_out}.summary.json");
    let v: serde_json::Value = mumdia_io::json::read_json(&path)
        .with_context(|| format!("reading {path} to check the deferred DeepLC pass"))?;
    let n = |k: &str| v.get(k).and_then(|x| x.as_u64());
    let retained = n("retained_imported")
        .ok_or_else(|| anyhow::anyhow!("{path} does not report retained_imported"))?;
    if retained > 0 {
        bail!(
            "the multi-head calibration kept the input iRT of {retained} of {} library rows \
             ({} with non-standard residues, {} without a finite prediction), and under \
             predict_frag.defer_deeplc_to_multihead that input is the native model's \
             placeholder, not a DeepLC prediction. Set predict_frag.defer_deeplc_to_multihead \
             = false to predict the library with DeepLC first; {path} has the counts",
            n("rows").unwrap_or(0),
            n("retained_non_standard").unwrap_or(0),
            n("retained_no_prediction").unwrap_or(0),
        );
    }
    info!(
        rows = n("rows").unwrap_or(0),
        library = lib_out,
        "sidecar: the multi-head calibration re-predicted every library row, so the deferred \
         DeepLC pass was not needed"
    );
    Ok(())
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
    // The worker sizes torch's CPU pool from this, capped at the physical cores available
    // to it (`MUMDIA_DEEPLC_THREAD_CAP`). Without it torch took its own default, which
    // ignored the engine's `--threads` (docs/13, "DeepLC thread cap").
    let threads = rayon::current_num_threads().max(1).to_string();
    info!(n = ids.len(), threads = %threads, "sidecar: running DeepLC");
    run_worker(python, script, &[&inp, &outp, &threads], true).context("DeepLC worker failed")?;
    read_id_scalars(
        &outp,
        ids,
        "DeepLC",
        "peptidoforms",
        "predicted_rt",
        TableFile::f32,
        false,
    )
}

/// IM2Deep: one predicted 1/K0 (V s cm^-2) per (peptidoform, charge), from the worker's
/// CCS converted at the given precursor m/z. Returns `id -> predicted_im`. Positional
/// contract: `im2deep_worker.py <in.parquet> <out.parquet> <threads>`; the request is the
/// fragment workers' `id, peptidoform, charge` table plus `precursor_mz`.
pub fn run_im2deep(
    python: &str,
    script: &str,
    workdir: &str,
    ids: &[u32],
    peptidoforms: &[String],
    charges: &[i32],
    precursor_mz: &[f64],
) -> Result<HashMap<u32, f64>> {
    require_version(
        python,
        "im2deep",
        mumdia_core::constants::MIN_IM2DEEP_VERSION,
    )?;
    let (inp, outp) = fragment_request(
        workdir,
        "im2deep",
        ids,
        peptidoforms,
        charges,
        Some(("precursor_mz", precursor_mz)),
    )?;
    // The worker caps torch's CPU pool at the physical cores available to it
    // (`MUMDIA_IM2DEEP_THREAD_CAP`), as the DeepLC workers do.
    let threads = rayon::current_num_threads().max(1).to_string();
    info!(n = ids.len(), threads = %threads, "sidecar: running IM2Deep");
    run_worker(python, script, &[&inp, &outp, &threads], true).context("IM2Deep worker failed")?;
    read_id_scalars(
        &outp,
        ids,
        "IM2Deep",
        "precursors",
        "predicted_im",
        TableFile::f64,
        true,
    )
}

/// DeepLC multitask fine-tune: adapt the RT model to this run's confident seed
/// PSMs and rewrite the supplied library's `predicted_irt`. Positional contract:
/// `deeplc_finetune.py <lib_in> <seed> <lib_out>`.
///
/// Training keeps the worker's bounded pool (`--threads`, default 8: the documented
/// OpenMP crash was the backward pass). The whole-library prediction after it is
/// forward-only and gets `threads` (`--predict-threads`), which the worker caps at the
/// physical cores available to it; before, it ran on the 8 training threads. On a host
/// with more than 8 physical cores that moves the output in the last bits whether or not
/// the cap binds, and `DEEPLC_FT_THREADS` now bounds training only (docs/13, "DeepLC
/// thread cap").
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
    threads: usize,
    shards: usize,
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
        threads,
        shards,
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
    let pt = threads.max(1).to_string();
    let mut args = vec![
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
        "--predict-threads",
        &pt,
    ];
    let sh = shards.to_string();
    push_shards(&mut args, shards, &sh);
    run_worker(python, script, &args, true).context("DeepLC fine-tune failed")?;
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
/// `deeplc_finetune.py <lib_in> <seed> <lib_out> --multihead <n_heads>`. `threads` sizes
/// both torch pools, and the worker caps it at the physical cores available to it.
/// `shards` is `rt_im_train.deeplc_predict_shards`.
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
    shards: usize,
    projection_cache: Option<&crate::cache::CacheDir>,
) -> Result<()> {
    require_deeplc_version(python)?;
    info!(
        lib_in,
        seed,
        lib_out,
        n_heads,
        q_train,
        threads,
        shards,
        "sidecar: calibrating DeepLC over multiple heads"
    );
    let nh = n_heads.to_string();
    let qt = q_train.to_string();
    let hf = window_holdout_frac.to_string();
    let th = threads.max(1).to_string();
    let mut args = vec![
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
    ];
    let sh = shards.to_string();
    push_shards(&mut args, shards, &sh);
    push_projection_cache(&mut args, projection_cache);
    run_worker(python, script, &args, true).context("DeepLC multi-head calibration failed")?;
    warn_on_retained_imported(lib_out);
    Ok(())
}

/// DeepLC base-model re-prediction of an imported library's `predicted_irt`: the
/// fine-tune worker with `--no-finetune`, so the table rewrite (targets predicted on their
/// peptidoform, decoys on the DECOY_-stripped sequence, rows with non-standard residues
/// keeping the imported value) is the one the fine-tune path uses. Positional contract:
/// `deeplc_finetune.py <lib_in> - <lib_out> --no-finetune`. Prediction is forward-only,
/// so it asks for the engine's full thread count rather than the fine-tune's bounded pool,
/// and the worker caps that at the physical cores available to it: past them a second
/// hyperthread on a busy core slows every OpenMP-parallel op (the multi-head step took
/// 10:41 at 96 threads and 18:09 at 128 on a 64-core host).
pub fn run_deeplc_repredict(
    python: &str,
    script: &str,
    lib_in: &str,
    lib_out: &str,
    threads: usize,
    shards: usize,
    projection_cache: Option<&crate::cache::CacheDir>,
) -> Result<()> {
    require_deeplc_version(python)?;
    info!(
        lib_in,
        lib_out,
        threads,
        shards,
        "sidecar: re-predicting the library iRT with the DeepLC base model"
    );
    let th = threads.max(1).to_string();
    let mut args = vec![
        lib_in,
        "-",
        lib_out,
        "--no-finetune",
        "--threads",
        &th,
        "--predict-threads",
        &th,
    ];
    let sh = shards.to_string();
    push_shards(&mut args, shards, &sh);
    push_projection_cache(&mut args, projection_cache);
    run_worker(python, script, &args, true).context("DeepLC library re-prediction failed")?;
    warn_on_retained_imported(lib_out);
    Ok(())
}

/// What a band-list DeepLC call ([`run_deeplc_bands`]) does to every band.
pub enum BandAdaptation<'a> {
    /// The multi-head calibration against `seed`, fitted once for all the bands.
    Multihead {
        seed: &'a str,
        n_heads: usize,
        q_train: f64,
        window_holdout_frac: f64,
    },
    /// The base-model re-prediction.
    Repredict,
}

/// One DeepLC worker for all the bands of a grouped run (`groups.rt_adaptation =
/// once_per_run`): `deeplc_finetune.py - <seed|-> - --bands <tsv>`, where `tsv` lists
/// `lib_in<TAB>lib_out` per band. The worker fits the calibration once, predicts the union
/// of the bands' unique sequences once, and writes each `lib_out` with the rewrite a
/// single-table call applies, each with its own `<lib_out>.summary.json`. `threads` and
/// `shards` mean what they mean for [`run_deeplc_multihead`].
#[allow(clippy::too_many_arguments)]
pub fn run_deeplc_bands(
    python: &str,
    script: &str,
    pairs: &[(String, String)],
    tsv: &str,
    mode: BandAdaptation,
    threads: usize,
    shards: usize,
    projection_cache: Option<&crate::cache::CacheDir>,
) -> Result<()> {
    require_deeplc_version(python)?;
    if pairs.is_empty() {
        bail!("no band to adapt");
    }
    let mut body = String::new();
    for (lib_in, lib_out) in pairs {
        if lib_in.contains(['\t', '\n', '\r']) || lib_out.contains(['\t', '\n', '\r']) {
            bail!("a band path contains a tab or a line break: {lib_in} -> {lib_out}");
        }
        body.push_str(lib_in);
        body.push('\t');
        body.push_str(lib_out);
        body.push('\n');
    }
    std::fs::write(tsv, body).with_context(|| format!("writing the band list {tsv}"))?;
    let th = threads.max(1).to_string();
    let sh = shards.to_string();
    let (nh, qt, hf);
    let mut args: Vec<&str> = Vec::new();
    match mode {
        BandAdaptation::Multihead {
            seed,
            n_heads,
            q_train,
            window_holdout_frac,
        } => {
            nh = n_heads.to_string();
            qt = q_train.to_string();
            hf = window_holdout_frac.to_string();
            info!(
                bands = pairs.len(),
                seed,
                n_heads,
                q_train,
                threads,
                shards,
                "sidecar: calibrating DeepLC over multiple heads once for every band"
            );
            args.extend_from_slice(&[
                "-",
                seed,
                "-",
                "--bands",
                tsv,
                "--multihead",
                &nh,
                "--q-train",
                &qt,
                "--window-holdout-frac",
                &hf,
            ]);
        }
        BandAdaptation::Repredict => {
            info!(
                bands = pairs.len(),
                threads,
                shards,
                "sidecar: re-predicting the band libraries' iRT with the DeepLC base model \
                 in one pass"
            );
            args.extend_from_slice(&["-", "-", "-", "--bands", tsv, "--no-finetune"]);
        }
    }
    args.extend_from_slice(&["--threads", &th, "--predict-threads", &th]);
    push_shards(&mut args, shards, &sh);
    push_projection_cache(&mut args, projection_cache);
    run_worker(python, script, &args, true).context("DeepLC band adaptation failed")?;
    for (_, lib_out) in pairs {
        if !std::path::Path::new(lib_out).exists() {
            bail!("the DeepLC band worker exited 0 but wrote no {lib_out}");
        }
        warn_on_retained_imported(lib_out);
    }
    Ok(())
}

/// Append `--projection-cache <dir>` (`rt_im_train.deeplc_projection_cache`), and
/// `--projection-cache-default` when the directory is the engine's default (`"auto"`,
/// `crate::cache`): the worker then reports in a plain line, not a warning, that a DeepLC
/// older than 4.5.0 cannot use it, since the default asks for the cache only where it is
/// available. Nothing when the cache is off.
fn push_projection_cache<'a>(args: &mut Vec<&'a str>, dir: Option<&'a crate::cache::CacheDir>) {
    let Some(dir) = dir else {
        return;
    };
    let Some(path) = dir.path.to_str() else {
        warn!(
            dir = %dir.path.display(),
            "rt_im_train.deeplc_projection_cache: the directory is not valid UTF-8; predicting \
             without the projection cache"
        );
        return;
    };
    args.push("--projection-cache");
    args.push(path);
    if dir.auto {
        args.push("--projection-cache-default");
    }
}

/// Append `--shards <n>` for a sharded whole-library prediction
/// (`rt_im_train.deeplc_predict_shards`). One process, the default, passes nothing, so the
/// default argument list is the one an older `deeplc_finetune.py` accepts.
fn push_shards<'a>(args: &mut Vec<&'a str>, shards: usize, rendered: &'a str) {
    if shards != 1 {
        args.push("--shards");
        args.push(rendered);
    }
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
        let diagnosis = failure_diagnosis(python, script);
        bail!(
            "sidecar worker failed: {script} exited with {status}.\n\
             interpreter: {python}\n\
             command: {python} {script} {argv}\n\
             The worker's own output, including any Python traceback, is above this \
             error. Reproduce it by running that command directly. \
             `mumdia doctor --config <config>` checks the environment.{diagnosis}"
        );
    }
    Ok(())
}

/// The torch-backed modules a worker imports first, which a failed run re-imports to
/// explain itself. Empty for a worker that never loads torch.
fn torch_modules_of(script: &str) -> &'static [&'static str] {
    let name = std::path::Path::new(script)
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if name.starts_with("deeplc") {
        &["deeplc", "torch"]
    } else if name.starts_with("im2deep") {
        &["im2deep", "torch"]
    } else if name.starts_with("peptdeep") {
        &["peptdeep", "torch"]
    } else if name.starts_with("nn_rescore") {
        &["torch"]
    } else {
        &[]
    }
}

/// What to add to the error of a failed torch-backed worker: the result of a real import
/// of what it imports, and the fix when that names one (`python::load_failure_hint`).
/// Empty when the imports load, so the error is unchanged for every other failure.
///
/// Worker output is inherited rather than captured (see `run_worker`), so the engine
/// never holds the worker's traceback; a fresh import in the same interpreter reproduces
/// a load failure deterministically and costs a few seconds only on a path that has
/// already failed. This is what reaches a user who never runs `doctor`: a torch whose
/// DLLs cannot initialise on Windows otherwise ends a search minutes in with an
/// `OSError: [WinError 1114]` and no hint that the Visual C++ runtime is the cause.
pub fn failure_diagnosis(python: &str, script: &str) -> String {
    let modules = torch_modules_of(script);
    if modules.is_empty() {
        return String::new();
    }
    let probed = crate::python::import_probe(
        std::path::Path::new(python),
        modules,
        crate::python::FAILURE_PROBE_TIMEOUT,
    );
    describe_failure_probe(probed.ok())
}

/// The text `failure_diagnosis` appends for a probe result; split out to be tested
/// without an interpreter.
fn describe_failure_probe(probe: Option<crate::python::ImportProbe>) -> String {
    match probe {
        Some(crate::python::ImportProbe::Unloadable { module, detail }) => {
            let hint = crate::python::load_failure_hint(&detail)
                .map(|h| format!("\n{h}"))
                .unwrap_or_default();
            format!(
                "\nA fresh `import {module}` in this interpreter fails as well, so the \
                 environment cannot load it:\n{detail}{hint}"
            )
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod failure_diagnosis_tests {
    use super::{describe_failure_probe, torch_modules_of};
    use crate::python::{ImportProbe, VC_RUNTIME_HINT};

    #[test]
    fn only_torch_backed_workers_are_probed_after_a_failure() {
        assert_eq!(
            torch_modules_of("C:/Program Files/MuMDIA/binaries/scripts/deeplc_finetune.py"),
            &["deeplc", "torch"]
        );
        assert_eq!(
            torch_modules_of("scripts/deeplc_worker.py"),
            &["deeplc", "torch"]
        );
        assert_eq!(torch_modules_of("scripts/nn_rescore_worker.py"), &["torch"]);
        assert_eq!(
            torch_modules_of("scripts/im2deep_worker.py"),
            &["im2deep", "torch"]
        );
        assert!(torch_modules_of("scripts/mbr_worker.py").is_empty());
        assert!(torch_modules_of("scripts/ms2pip_worker.py").is_empty());
    }

    #[test]
    fn a_dll_failure_adds_the_traceback_and_the_runtime_hint() {
        let detail = "OSError: [WinError 1114] A dynamic link library (DLL) initialization \
                      routine failed. Error loading \"C:\\env\\torch\\lib\\c10.dll\""
            .to_string();
        let text = describe_failure_probe(Some(ImportProbe::Unloadable {
            module: "deeplc".into(),
            detail: detail.clone(),
        }));
        assert!(text.contains("`import deeplc`"));
        assert!(text.contains(&detail));
        assert!(text.ends_with(VC_RUNTIME_HINT));
        // Another import failure is reported without a hint that does not apply.
        let text = describe_failure_probe(Some(ImportProbe::Unloadable {
            module: "torch".into(),
            detail: "ImportError: numpy.core.multiarray failed to import".into(),
        }));
        assert!(text.contains("numpy.core.multiarray"));
        assert!(!text.contains("Visual C++"));
        // Imports that load, a hung probe and an interpreter that cannot start add nothing.
        assert_eq!(describe_failure_probe(Some(ImportProbe::Loaded)), "");
        assert_eq!(
            describe_failure_probe(Some(ImportProbe::TimedOut { seconds: 60 })),
            ""
        );
        assert_eq!(describe_failure_probe(None), "");
    }
}

#[cfg(test)]
mod placeholder_tests {
    use super::require_every_row_repredicted;

    #[test]
    fn a_deferred_build_fails_when_the_calibration_kept_a_row() {
        let dir = std::env::temp_dir().join(format!("mumdia_placeholder_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lib = dir.join("lib.parquet").to_str().unwrap().to_string();
        let summary = format!("{lib}.summary.json");
        std::fs::write(
            &summary,
            r#"{"rows": 10, "repredicted": 10, "retained_imported": 0}"#,
        )
        .unwrap();
        require_every_row_repredicted(&lib).unwrap();
        std::fs::write(
            &summary,
            r#"{"rows": 10, "repredicted": 9, "retained_imported": 1,
                "retained_non_standard": 0, "retained_no_prediction": 1}"#,
        )
        .unwrap();
        let e = require_every_row_repredicted(&lib).unwrap_err().to_string();
        assert!(e.contains("1 of 10"), "{e}");
        assert!(e.contains("defer_deeplc_to_multihead"), "{e}");
        // No summary, or one without the count, is not a pass.
        std::fs::write(&summary, r#"{"rows": 10}"#).unwrap();
        assert!(require_every_row_repredicted(&lib).is_err());
        std::fs::remove_file(&summary).unwrap();
        assert!(require_every_row_repredicted(&lib).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod scalar_reader_tests {
    use super::{fragment_request, read_id_scalars};
    use mumdia_io::table::{write_table, Col, TableFile};

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_scalars_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn table_f64(dir: &std::path::Path, name: &str, ids: Vec<u32>, v: Vec<f64>) -> String {
        let p = dir
            .join(format!("{name}.parquet"))
            .to_string_lossy()
            .into_owned();
        write_table(
            &p,
            vec![
                Col::U32("id".into(), ids),
                Col::F64("predicted_im".into(), v),
            ],
        )
        .unwrap();
        p
    }

    fn im(path: &str, ids: &[u32]) -> anyhow::Result<std::collections::HashMap<u32, f64>> {
        read_id_scalars(
            path,
            ids,
            "IM2Deep",
            "precursors",
            "predicted_im",
            TableFile::f64,
            true,
        )
    }

    #[test]
    fn the_reader_checks_the_subset_each_id_once_and_optionally_the_value() {
        let dir = scratch("checks");
        // A subset of the request is fine: coverage is the caller's to judge.
        let ok = table_f64(&dir, "ok", vec![0, 2], vec![0.8, 1.1]);
        let m = im(&ok, &[0, 1, 2]).unwrap();
        assert_eq!((m[&0], m[&2], m.len()), (0.8, 1.1, 2));
        let stray = table_f64(&dir, "stray", vec![0, 9], vec![0.8, 1.1]);
        let e = im(&stray, &[0, 1]).unwrap_err().to_string();
        assert!(
            e.contains("id 9") && e.contains("2 precursors requested"),
            "{e}"
        );
        let dup = table_f64(&dir, "dup", vec![0, 0], vec![0.8, 0.9]);
        assert!(im(&dup, &[0])
            .unwrap_err()
            .to_string()
            .contains("more than once"));
        for (name, bad) in [
            ("nan", f64::NAN),
            ("zero", 0.0),
            ("neg", -0.5),
            ("inf", f64::INFINITY),
        ] {
            let p = table_f64(&dir, name, vec![0], vec![bad]);
            assert!(im(&p, &[0]).is_err(), "{name}");
            // The same value passes with the check off, as DeepLC's readback runs.
            let off =
                read_id_scalars(&p, &[0], "x", "y", "predicted_im", TableFile::f64, false).unwrap();
            assert_eq!(off.len(), 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_deeplc_readback_keeps_its_values_and_its_messages() {
        // `run_deeplc` reads through the shared reader with the check off: a negative or
        // zero retention time on the model's scale comes back unchanged, bit for bit.
        let dir = scratch("deeplc");
        let p = dir.join("out.parquet").to_string_lossy().into_owned();
        let rt = vec![-3.25f32, 0.0, 1234.567];
        write_table(
            &p,
            vec![
                Col::U32("id".into(), vec![0, 1, 2]),
                Col::F32("predicted_rt".into(), rt.clone()),
            ],
        )
        .unwrap();
        let read = |ids: &[u32]| {
            read_id_scalars(
                &p,
                ids,
                "DeepLC",
                "peptidoforms",
                "predicted_rt",
                TableFile::f32,
                false,
            )
        };
        let m = read(&[0, 1, 2]).unwrap();
        for (i, v) in rt.iter().enumerate() {
            assert_eq!(m[&(i as u32)].to_bits(), v.to_bits());
        }
        let e = read(&[0, 1]).unwrap_err().to_string();
        assert_eq!(
            e,
            "DeepLC worker returned id 2, which was not among the 2 peptidoforms requested"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_request_gains_an_extra_column_only_when_asked() {
        let dir = scratch("request");
        let w = dir.to_string_lossy().into_owned();
        let peps = vec!["PEPTIDEK".to_string(), "SAMPLER".to_string()];
        let (inp, _) = fragment_request(&w, "plain", &[0, 1], &peps, &[2, 3], None).unwrap();
        assert_eq!(
            mumdia_io::table::column_names(&inp).unwrap(),
            ["id", "peptidoform", "charge"]
        );
        let (inp, _) = fragment_request(
            &w,
            "mz",
            &[0, 1],
            &peps,
            &[2, 3],
            Some(("precursor_mz", &[450.5, 401.2])),
        )
        .unwrap();
        let t = TableFile::open(&inp).unwrap();
        assert_eq!(t.f64("precursor_mz").unwrap(), vec![450.5, 401.2]);
        assert_eq!(t.i32("charge").unwrap(), vec![2, 3]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod shard_arg_tests {
    use super::push_shards;

    #[test]
    fn one_shard_adds_nothing_so_an_older_worker_still_parses_the_call() {
        let mut args = vec!["lib_in", "-", "lib_out"];
        push_shards(&mut args, 1, "1");
        assert_eq!(args, ["lib_in", "-", "lib_out"]);
    }

    #[test]
    fn several_or_automatic_shards_are_passed_through() {
        for (n, rendered) in [(4usize, "4"), (0, "0")] {
            let mut args = vec!["lib_in"];
            push_shards(&mut args, n, rendered);
            assert_eq!(args, ["lib_in", "--shards", rendered]);
        }
    }
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
