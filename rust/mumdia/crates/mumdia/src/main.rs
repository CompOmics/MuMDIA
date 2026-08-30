//! MuMDIA CLI: one binary, one subcommand per stage (docs/01_overview_and_dataflow.md).
//! Every stage runs standalone on path-addressable inputs.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use mumdia::python;
use mumdia::stages;
use mumdia_core::config::Config;

/// Per-thread-arena allocator. The extraction accumulation allocates heavily inside
/// rayon workers and measured only ~1.07x parallel scaling under the Windows system
/// allocator's shared heap lock. Swapping the allocator changes no results.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(
    name = "mumdia",
    version,
    about = "MuMDIA DIA search engine (Rust MVP)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,

    /// Maximum worker threads. Default: every core.
    ///
    /// Bounds the engine's rayon pool and is forwarded to the Python sidecars as
    /// `MUMDIA_NN_THREADS` and `OMP_NUM_THREADS` unless those are already set.
    /// Without this there was no way to bound MuMDIA at all except the
    /// undocumented `RAYON_NUM_THREADS`, which the engine never read and which
    /// does not reach the sidecars; on a shared machine that made a run
    /// antisocial. Note the NN rescore worker measured FASTER on 8 threads than
    /// on 32 (docs/13_sidecars.md).
    #[arg(long, global = true, value_name = "N")]
    threads: Option<usize>,

    /// Log level: `error`, `warn`, `info` (default), `debug`, or `trace`. Accepts
    /// any `RUST_LOG` filter, so `mumdia=debug,extract=trace` also works.
    #[arg(long, global = true, value_name = "LEVEL")]
    log_level: Option<String>,

    /// More detail: `-v` for debug, `-vv` for trace. Overridden by --log-level.
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Warnings and errors only. Overridden by --log-level.
    #[arg(short = 'q', long, global = true, conflicts_with = "verbose")]
    quiet: bool,
}

impl Cli {
    /// The tracing filter these flags ask for, or `None` to leave `RUST_LOG` in
    /// charge. `--log-level` is explicit and wins; otherwise the counted `-v` and
    /// `-q` map onto levels.
    fn log_filter(&self) -> Option<String> {
        if let Some(l) = &self.log_level {
            return Some(l.clone());
        }
        match (self.quiet, self.verbose) {
            (true, _) => Some("warn".into()),
            (false, 0) => None,
            (false, 1) => Some("debug".into()),
            (false, _) => Some("trace".into()),
        }
    }
}

/// Apply `--threads` to the engine's own pool and to the sidecars.
///
/// Rayon's global pool can only be built once and only before first use, so this
/// runs before any subcommand. An existing environment variable is left alone: a
/// user who set `OMP_NUM_THREADS` for a reason should not have it silently
/// replaced.
fn apply_threads(threads: Option<usize>) -> Result<()> {
    let Some(n) = threads else { return Ok(()) };
    if n == 0 {
        anyhow::bail!("--threads must be >= 1");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| anyhow::anyhow!("cannot set --threads {n}: {e}"))?;
    for var in ["MUMDIA_NN_THREADS", "OMP_NUM_THREADS"] {
        if std::env::var_os(var).is_none() {
            std::env::set_var(var, n.to_string());
        }
    }
    tracing::info!(threads = n, "threads: engine pool and sidecar hints set");
    Ok(())
}

#[derive(Subcommand)]
enum Cmd {
    /// Read an mzML run into the normalized spectra artifact set.
    Convert {
        #[arg(long)]
        mzml: String,
        #[arg(long)]
        out_dir: String,
        /// Limit spectra read (0 = all), for fast iteration.
        #[arg(long, default_value_t = 0)]
        max_spectra: usize,
        /// Keep at most this many MS2 peaks in the normalized artifact (0 = all).
        ///
        /// This is an irreversible conversion-time cap that also affects extraction,
        /// features, and quantification. Use `search_seed.top_n_peaks` for a
        /// seed-only limit.
        #[arg(long, default_value_t = 0)]
        top_peaks_ms2: usize,
        /// Keep at most this many MS1 peaks per scan (0 = all).
        #[arg(long, default_value_t = 0)]
        top_peaks_ms1: usize,
    },
    /// Fully-tryptic digest + decoy pairing -> peptides.parquet.
    Digest {
        #[arg(long)]
        fasta: String,
        #[arg(long)]
        out: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Fixed+variable modification and charge enumeration -> peptidoforms.parquet.
    Peptidoforms {
        #[arg(long)]
        peptides: String,
        #[arg(long)]
        out: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Spectral library: b/y m/z + predicted intensity + iRT -> fragment_library.
    PredictFrag {
        #[arg(long)]
        peptidoforms: String,
        #[arg(long)]
        out_precursors: String,
        #[arg(long)]
        out_fragments: String,
        /// Working directory for sidecar request/response files.
        #[arg(long, default_value = "sidecar_work")]
        work_dir: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Sequence-tag prescan: keep only modification-bearing candidates whose anchored trimers are
    /// observed in this run -> prescan_survivors.parquet. Label-blind by construction, so it
    /// prunes search space without touching target-decoy exchangeability.
    Prescan {
        #[arg(long)]
        ms2: String,
        #[arg(long)]
        isolation_windows: String,
        #[arg(long)]
        lib_precursors: String,
        /// Per-candidate RT bounds (candidate_id, rt_lo, rt_hi); a run_windows-shaped table.
        #[arg(long)]
        run_windows: String,
        #[arg(long)]
        out: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Native broad DIA seed search over the fragment index -> seed_psms.parquet.
    SearchSeed {
        #[arg(long)]
        ms2: String,
        #[arg(long)]
        lib_precursors: String,
        #[arg(long)]
        lib_fragments: String,
        #[arg(long)]
        out: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Per-run RT calibration + windows -> run_windows.parquet, cal.json.
    RtImTrain {
        #[arg(long)]
        seed_psms: String,
        #[arg(long)]
        lib_precursors: String,
        #[arg(long)]
        out_windows: String,
        #[arg(long)]
        out_cal: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Targeted 3D extraction (peak-major cascade) -> psms_extracted, chromatograms.
    Extract {
        #[arg(long)]
        ms2: String,
        #[arg(long)]
        lib_precursors: String,
        #[arg(long)]
        lib_fragments: String,
        #[arg(long)]
        run_windows: String,
        /// Optional MS1 spectra for isotope-envelope features.
        #[arg(long)]
        ms1: Option<String>,
        /// Optional mass recalibration json (search-seed <seed>.masscal.json).
        #[arg(long)]
        mass_cal: Option<String>,
        #[arg(long)]
        out_psms: String,
        #[arg(long)]
        out_chromatograms: String,
        /// Optional candidate allowlist (a prior run's psms.parquet): restrict
        /// extraction to these candidate_ids. For "gate first, then compete" -
        /// re-extract with a peak_claim strategy over only the gate-accepted
        /// survivors, keeping the two-pass profile map small.
        #[arg(long)]
        restrict_candidates: Option<String>,
        #[arg(long)]
        config: Option<String>,
    },
    /// Compute the minimal feature set -> features.parquet + PIN.
    Features {
        #[arg(long)]
        psms_extracted: String,
        #[arg(long)]
        chromatograms: String,
        /// Optional seed_psms for search-engine corroboration features.
        #[arg(long)]
        seed_psms: Option<String>,
        #[arg(long)]
        out: String,
        #[arg(long)]
        out_pin: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Keep the best candidate per competition group -> psms_competed.parquet.
    Compete {
        #[arg(long)]
        features: String,
        #[arg(long)]
        out: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Rescore + native target-decoy q-values -> psms_scored.parquet.
    Rescore {
        /// One or more competed feature tables.
        #[arg(long, num_args = 1..)]
        competed: Vec<String>,
        #[arg(long)]
        out: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Quantify identified peptides + roll up to protein groups.
    Quant {
        #[arg(long)]
        psms_scored: String,
        #[arg(long)]
        chromatograms: String,
        #[arg(long)]
        out_peptide: String,
        #[arg(long)]
        out_protein: String,
        /// Optional per-fragment area export (for ion-level directLFQ).
        #[arg(long)]
        out_fragment: Option<String>,
        /// Optional per-candidate peak-window diagnostic (candidate_id, lo_rt, hi_rt, width_s).
        #[arg(long)]
        out_peak_bounds: Option<String>,
        #[arg(long)]
        config: Option<String>,
    },
    /// Combine per-run quant tables into a protein-by-run matrix (cross-run LFQ).
    QuantLfq {
        /// One per-run table per run: peptide_quant.parquet for maxlfq,
        /// fragment_quant.parquet for directlfq.
        #[arg(long, num_args = 1..)]
        inputs: Vec<String>,
        /// `maxlfq` (peptide-level) or `directlfq` (ion/fragment-level).
        #[arg(long, default_value = "maxlfq")]
        method: String,
        /// Cross-run normalization: `median_ratio` (default), `median`, or `none`.
        #[arg(long, default_value = "median_ratio")]
        normalize: String,
        #[arg(long)]
        out: String,
    },
    /// Orchestrate the full MVP pipeline on one run and write a manifest.
    Run {
        /// FASTA to digest into the library. Omit when supplying a prebuilt
        /// library via --lib-precursors + --lib-fragments (library-input mode).
        #[arg(long)]
        fasta: Option<String>,
        #[arg(long)]
        mzml: String,
        #[arg(long)]
        out_dir: String,
        /// Library-input mode: consume a prebuilt precursor library (e.g. an
        /// imported DIA-NN speclib) instead of digesting --fasta. Requires
        /// --lib-fragments; skips digest/peptidoforms/predict-frag.
        #[arg(long)]
        lib_precursors: Option<String>,
        /// Prebuilt fragment library paired with --lib-precursors.
        #[arg(long)]
        lib_fragments: Option<String>,
        #[arg(long)]
        config: Option<String>,
        /// Named tuning preset applied on top of --config/defaults. "dia" = the
        /// validated DIA preset (Extended features, rolling-window apex, RT prior).
        #[arg(long)]
        profile: Option<String>,
        #[arg(long, default_value_t = 0)]
        max_spectra: usize,
        /// Irreversible conversion-time MS2 cap (0 = all). Seed-only peak limiting
        /// is configured by `search_seed.top_n_peaks`.
        #[arg(long, default_value_t = 0)]
        top_peaks_ms2: usize,
    },
    /// Experiment-wide orchestrator: run the per-file search chain over N runs,
    /// then one combined rescore, optional rescuable MBR transfer, per-run quant,
    /// and cross-run LFQ. Pass --mzml once per run (>= 2).
    RunExperiment {
        #[arg(long)]
        fasta: Option<String>,
        /// One per run; repeat the flag (>= 2 runs).
        #[arg(long)]
        mzml: Vec<String>,
        /// Optional per-run labels / subdir names (default r0..rN-1).
        #[arg(long)]
        run_names: Vec<String>,
        #[arg(long)]
        out_dir: String,
        #[arg(long)]
        lib_precursors: Option<String>,
        #[arg(long)]
        lib_fragments: Option<String>,
        #[arg(long)]
        config: Option<String>,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long, default_value_t = 0)]
        max_spectra: usize,
        #[arg(long, default_value_t = 0)]
        top_peaks_ms2: usize,
    },
    /// Cross-run RT alignment (experiment-level) -> alignment.parquet.
    Align {
        /// One seed_psms.parquet per run; the first is the reference.
        #[arg(long, num_args = 1..)]
        seed_psms: Vec<String>,
        #[arg(long)]
        out: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Match-between-runs identification transfer (Stage D3) -> transferred.parquet.
    Mbr {
        /// Experiment-wide scored_combined.parquet (has the `source` column).
        #[arg(long)]
        psms_scored: String,
        /// Per-run psms.parquet in `source` order (one per run).
        #[arg(long, num_args = 1..)]
        psms_extracted: Vec<String>,
        #[arg(long)]
        out: String,
        /// Optional augmented scored table: input scored with accepted transfers'
        /// q_value lowered + is_transferred flag (for quant/report with q_filter=psm_q).
        #[arg(long)]
        out_psms_scored: Option<String>,
        /// Optional per-run fragment_quant.parquet (source order) for the
        /// fragment-consensus guard (needs mbr.consensus_corr_min > 0).
        #[arg(long, num_args = 0..)]
        frag: Vec<String>,
        #[arg(long)]
        config: Option<String>,
    },
    /// Print schema, head sample, and row count for any artifact.
    Inspect { artifact: String },
    /// Candidate audit: reconstruct per-candidate stage flags + earliest rejection
    /// reason across the artifact chain and write candidate_audit.parquet
    /// (sensitivity program, P0.3/P0.4). Non-destructive; reruns no compute.
    Audit {
        /// Library precursors parquet (the full candidate search space).
        #[arg(long)]
        lib_precursors: String,
        /// psms parquet from `extract`.
        #[arg(long)]
        psms_extracted: String,
        /// competed parquet from `compete`.
        #[arg(long)]
        competed: String,
        /// scored parquet from `rescore`.
        #[arg(long)]
        psms_scored: String,
        /// Output candidate_audit.parquet.
        #[arg(long)]
        out: String,
        /// Precursor q-value threshold for passed_precursor_fdr / reported.
        #[arg(long, default_value_t = 0.01)]
        q: f64,
        /// Run identifier stamped on every row.
        #[arg(long, default_value = "run")]
        run_id: String,
        /// Optional protein substring marking entrapment candidates (e.g. _HUMAN).
        #[arg(long, default_value = "")]
        entrapment_substr: String,
    },
    /// Write peptides.tsv + proteins.tsv from a scored PSM table.
    Report {
        #[arg(long)]
        psms_scored: String,
        #[arg(long)]
        out_dir: String,
        #[arg(long)]
        peptide_quant: Option<String>,
        #[arg(long)]
        protein_quant: Option<String>,
        /// Reported q threshold. Defaults to `quant.q_threshold` from `--config`
        /// when that is given, otherwise 0.01. An explicit value always wins.
        #[arg(long)]
        q: Option<f64>,
        /// Read `quant.q_threshold` from this config, so a standalone report uses the
        /// same threshold as the `run` that produced the table.
        ///
        /// Without it, a config setting `quant.q_threshold = 0.05` yielded 0.05 from
        /// `run` and 0.01 from `report` on the same scored table, silently.
        #[arg(long)]
        config: Option<String>,
    },
    /// Check that the configured Python sidecar environments are usable.
    Doctor {
        #[arg(long)]
        config: Option<String>,
        /// Emit the report as JSON on stdout instead of prose on stdout.
        ///
        /// For a caller that has to act on the result rather than read it: the
        /// desktop application renders one row per role and offers to install what
        /// is missing, which means it needs the modules and versions as data, not a
        /// paragraph to regex. The exit status is unchanged.
        #[arg(long)]
        json: bool,
    },
}

/// One sidecar role, as `doctor` found it.
#[derive(serde::Serialize)]
struct RoleReport {
    /// `rescore` | `deeplc` | `ms2pip` | `mbr`
    role: String,
    /// The configuration field that names this interpreter.
    field: String,
    /// Does THIS configuration need the role at all.
    required: bool,
    /// `ok` | `fail` | `skip` | `warn`
    status: String,
    python: Option<String>,
    /// `configured`, an environment variable name, `CONDA_PREFIX`, `PATH`, ...
    provenance: String,
    /// What the workers for this role import.
    modules: Vec<String>,
    /// Of those, the ones this interpreter cannot import.
    missing: Vec<String>,
    /// Versions of the packages whose version changes results.
    versions: std::collections::BTreeMap<String, String>,
    /// The environment variable that overrides this role.
    env_var: String,
    /// Anything worth saying that is not a failure, such as a DeepLC below the floor.
    warnings: Vec<String>,
}

/// The worker-script directory check, which runs before any interpreter.
#[derive(serde::Serialize)]
struct ScriptsReport {
    /// `ok` | `fail` | `skip`
    status: String,
    dir: String,
    /// Worker files the directory should contain but does not.
    missing: Vec<String>,
    /// False when the configuration needs no sidecar, in which case the directory
    /// is never opened and its absence is not a problem.
    needed: bool,
}

/// The whole report. `ok` is the same verdict the exit status carries.
#[derive(serde::Serialize)]
struct DoctorReport {
    ok: bool,
    scripts: ScriptsReport,
    roles: Vec<RoleReport>,
}

/// Report whether this configuration can actually run: which interpreter each
/// sidecar role resolves to, whether it can import what its workers import, which
/// versions it has, and whether the worker scripts are where the engine will look
/// for them.
///
/// What this replaces: the previous version probed three hard-coded interpreters
/// and reported `[skip]` for anything unset. It never probed `mbr.python`, never
/// checked that `sidecar_script_dir` existed (the most common misconfiguration,
/// and the one baked into the tracked example config), and reported no versions,
/// so a DeepLC old enough to change results looked identical to a current one.
fn doctor_report(cfg: &Config, config_path: Option<&str>) -> DoctorReport {
    use mumdia::python::{self, Role, ALL_ROLES};

    let mut cfg = cfg.clone();
    let script_dir = python::resolve_script_dir(&cfg.predict_frag.sidecar_script_dir, config_path);
    let dir_moved = script_dir != cfg.predict_frag.sidecar_script_dir;
    cfg.predict_frag.sidecar_script_dir = script_dir.clone();

    let mut ok = true;
    let any_sidecar = ALL_ROLES.iter().any(|r| r.required_by(&cfg));

    // 1. Worker scripts, checked before the interpreters because a missing script
    //    directory makes every interpreter irrelevant. Skipped entirely when the
    //    configuration needs no sidecar: the native predictors and `native_tda`
    //    rescorer are the default, and that run must not be failed for a directory
    //    it never opens.
    let dir = std::path::Path::new(&script_dir);
    let scripts = if !any_sidecar {
        ScriptsReport {
            status: "skip".into(),
            dir: script_dir.clone(),
            missing: Vec::new(),
            needed: false,
        }
    } else if !dir.is_dir() {
        ok = false;
        ScriptsReport {
            status: "fail".into(),
            dir: script_dir.clone(),
            missing: Vec::new(),
            needed: true,
        }
    } else {
        let mut missing: Vec<String> = Vec::new();
        for role in ALL_ROLES {
            if !role.required_by(&cfg) {
                continue;
            }
            for worker in role.workers() {
                if !dir.join(worker).exists() {
                    missing.push(worker.to_string());
                }
            }
        }
        missing.sort_unstable();
        missing.dedup();
        if !missing.is_empty() {
            ok = false;
        }
        ScriptsReport {
            status: if missing.is_empty() { "ok" } else { "fail" }.into(),
            dir: script_dir.clone(),
            missing,
            needed: true,
        }
    };
    let _ = dir_moved;

    // 2. Interpreters, one entry per role, resolving `auto` exactly as a run would.
    let mut roles = Vec::new();
    for role in ALL_ROLES {
        let configured = match role {
            Role::Rescore => cfg.rescore.python.clone(),
            Role::DeepLc => cfg.predict_frag.deeplc_python.clone(),
            Role::Ms2pip => cfg.predict_frag.ms2pip_python.clone(),
            Role::Mbr => cfg.mbr.python.clone(),
        };
        let required = role.required_by(&cfg);
        let modules: Vec<String> = role.modules(&cfg).iter().map(|m| m.to_string()).collect();
        let explicit = configured
            .as_deref()
            .map(|v| !v.eq_ignore_ascii_case(python::AUTO))
            .unwrap_or(false);

        let mut r = RoleReport {
            role: format!("{role:?}").to_lowercase(),
            field: role.field().to_string(),
            required,
            status: "skip".into(),
            python: None,
            provenance: "not required".into(),
            modules,
            missing: Vec::new(),
            versions: std::collections::BTreeMap::new(),
            env_var: role.env_var().to_string(),
            warnings: Vec::new(),
        };

        // Neither needed nor named: say so and probe nothing. Discovery here used to
        // run for every role and then report the interpreter it happened to find as
        // "configured but not needed", which described neither the config nor the
        // outcome.
        if !required && !explicit {
            roles.push(r);
            continue;
        }

        let (path, provenance) = if explicit {
            (configured.clone(), "configured")
        } else {
            match python::discover(role, &cfg) {
                Some((p, src)) => (Some(p), src),
                None => (None, "not found"),
            }
        };
        r.provenance = provenance.to_string();
        r.python = path.clone();

        match (&path, required) {
            (None, true) => {
                ok = false;
                r.status = "fail".into();
            }
            (None, false) => r.status = "skip".into(),
            (Some(p), _) => {
                let interp = std::path::Path::new(p);
                let module_refs: Vec<&str> = r.modules.iter().map(|s| s.as_str()).collect();
                match python::missing_modules(interp, &module_refs) {
                    Ok(missing) if missing.is_empty() => {
                        for m in ["deeplc", "torch", "mokapot", "ms2pip", "numpy"] {
                            if module_refs.contains(&m) {
                                if let Some(v) = python::module_version(interp, m) {
                                    r.versions.insert(m.to_string(), v);
                                }
                            }
                        }
                        r.status = if required { "ok" } else { "note" }.into();
                        // DeepLC below 4.1.1 changes results rather than only
                        // performance: the 4.0.0a2 multitask preview overfits per-run
                        // fine-tuning badly enough to invert RT-model rankings
                        // (docs/08_rt_im_train.md).
                        if role == Role::DeepLc && required {
                            if let Some(v) = r.versions.get("deeplc") {
                                if version_below(v, &[4, 1, 1]) {
                                    r.warnings.push(format!(
                                        "DeepLC {v} is older than the supported floor 4.1.1; \
                                         results, not just speed, differ"
                                    ));
                                }
                            }
                        }
                    }
                    Ok(missing) => {
                        if required {
                            ok = false;
                        }
                        r.status = if required { "fail" } else { "warn" }.into();
                        r.missing = missing;
                    }
                    Err(e) => {
                        if required {
                            ok = false;
                        }
                        r.status = if required { "fail" } else { "warn" }.into();
                        r.warnings.push(e.to_string());
                    }
                }
            }
        }
        roles.push(r);
    }

    DoctorReport { ok, scripts, roles }
}

/// Render the report as the prose a person reads in a terminal.
fn print_doctor(rep: &DoctorReport) {
    println!("worker scripts");
    match rep.scripts.status.as_str() {
        "skip" => println!("  [skip] no Python sidecar is needed by this configuration"),
        "ok" => println!("  [ ok ] {}", rep.scripts.dir),
        _ if !rep.scripts.missing.is_empty() => println!(
            "  [FAIL] {}: missing {}",
            rep.scripts.dir,
            rep.scripts.missing.join(", ")
        ),
        _ => println!(
            "  [FAIL] predict_frag.sidecar_script_dir: {} is not a directory.\n\
             \x20        Point it at the `scripts/` directory that ships beside the binary.",
            rep.scripts.dir
        ),
    }

    println!("sidecar interpreters");
    for r in &rep.roles {
        let label = &r.field;
        match (r.status.as_str(), &r.python) {
            ("skip", _) => println!("  [skip] {label}: not needed by this config"),
            ("fail", None) => println!(
                "  [FAIL] {label}: required by this config, and no usable interpreter was \
                 found.\n\x20        Set it, set {}, or activate an environment that can \
                 import {}.",
                r.env_var,
                r.modules.join(", ")
            ),
            (status, Some(p))
                if r.missing.is_empty() && r.warnings.iter().all(|w| w.contains("DeepLC")) =>
            {
                let tag = if status == "ok" { " ok " } else { "note" };
                let notes: Vec<String> =
                    r.versions.iter().map(|(k, v)| format!("{k} {v}")).collect();
                println!(
                    "  [{tag}] {label}: {p} ({}){}",
                    r.provenance,
                    if notes.is_empty() {
                        String::new()
                    } else {
                        format!("\n\x20        {}", notes.join(", "))
                    }
                );
                if !r.required {
                    println!("\x20        (configured but not needed by this config)");
                }
                for w in &r.warnings {
                    println!("\x20        [warn] {w}");
                }
            }
            (status, Some(p)) if !r.missing.is_empty() => println!(
                "  [{}] {label}: {p} ({}) cannot import {}",
                if status == "fail" { "FAIL" } else { "warn" },
                r.provenance,
                r.missing.join(", ")
            ),
            (status, _) => {
                for w in &r.warnings {
                    println!(
                        "  [{}] {label}: {w}",
                        if status == "fail" { "FAIL" } else { "warn" }
                    );
                }
            }
        }
    }

    if rep.ok {
        println!("mumdia doctor: configuration is runnable");
    }
}

/// True when the dotted version `v` is below `floor`. Unparseable components
/// compare as 0, so a pre-release such as `4.0.0a2` reads as 4.0.0 and stays below
/// a 4.1.1 floor, which is the direction that matters here.
fn version_below(v: &str, floor: &[u32]) -> bool {
    let parts: Vec<u32> = v
        .split(['.', '-', '+'])
        .map(|p| {
            p.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .unwrap_or(0)
        })
        .collect();
    for (i, want) in floor.iter().enumerate() {
        let got = parts.get(i).copied().unwrap_or(0);
        if got != *want {
            return got < *want;
        }
    }
    false
}

/// Parse a config file without touching the sidecar interpreter fields.
///
/// Only `doctor` wants this. Its whole job is to report what a configuration would
/// resolve to and why, including when resolution would fail, so it must see the file
/// as written.
fn load_config_raw(path: &Option<String>) -> Result<Config> {
    match path {
        Some(p) => {
            let s = std::fs::read_to_string(p).with_context(|| {
                format!("failed to read config file {p} (check the path and spelling)")
            })?;
            Config::from_json(&s).with_context(|| format!("invalid config in {p}"))
        }
        None => Ok(Config::default()),
    }
}

/// Parse a config file and resolve everything path-shaped in it: the sidecar
/// interpreters (including the `"auto"` sentinel) and the worker script directory.
///
/// Every subcommand except `doctor` goes through here, which is the point. Resolution
/// used to happen only inside the two orchestrators, so `mumdia rescore`,
/// `mumdia predict-frag` and `mumdia mbr` took the word "auto" literally and tried to
/// execute a program by that name — while all three shipped example configs, and the
/// documented standalone-stage workflow, use `"auto"`. Resolving here means one
/// behaviour for every entry point.
///
/// Cost on the native path is nil: `python::resolve` probes only the roles that
/// `Role::required_by` says this configuration actually uses, and the default
/// configuration uses none.
fn load_config(path: &Option<String>) -> Result<Config> {
    let mut cfg = load_config_raw(path)?;
    cfg.predict_frag.sidecar_script_dir =
        python::resolve_script_dir(&cfg.predict_frag.sidecar_script_dir, path.as_deref());
    // BEST EFFORT, not Require. `Role::required_by` answers "does this configuration use
    // the role", which is not the same question as "does the subcommand I am about to run
    // use it": a `mumdia search-seed` with `rescore.classifier = entrapment` would
    // otherwise fail at config load demanding a mokapot interpreter for a rescorer that
    // stage never invokes. The two orchestrators call `python::resolve` themselves, and
    // keep the strict behaviour, which is where early failure actually pays.
    python::resolve_with(&mut cfg, python::Strictness::BestEffort)?;
    Ok(cfg)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    mumdia_io::init_logging_level(cli.log_filter().as_deref());
    apply_threads(cli.threads)?;
    match cli.cmd {
        Cmd::Convert {
            mzml,
            out_dir,
            max_spectra,
            top_peaks_ms2,
            top_peaks_ms1,
        } => {
            let cfg = load_config(&None)?;
            // Fold the conversion CLI caps into the convert artifacts' provenance
            // key: they change the spectra output but are not part of the config, so
            // two different caps would otherwise produce an identical config_hash
            // (docs/18_findings_and_decisions.md). The caps are also recorded in
            // the convert report.
            let config_hash = mumdia_io::hash::blake3_str(&format!(
                "{}\u{1f}max_spectra={max_spectra}\u{1f}top_peaks_ms2={top_peaks_ms2}\u{1f}top_peaks_ms1={top_peaks_ms1}",
                cfg.canonical_json()
            ));
            stages::convert::run(stages::convert::ConvertParams {
                mzml: &mzml,
                out_dir: &out_dir,
                max_spectra,
                top_peaks_ms2,
                top_peaks_ms1,
                config_hash: &config_hash,
            })?;
        }
        Cmd::Digest { fasta, out, config } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::digest::run(stages::digest::DigestParams {
                fasta: &fasta,
                out: &out,
                cfg: &cfg.digest,
                rng_seed: cfg.rng_seed,
                config_hash: &ch,
            })?;
        }
        Cmd::Peptidoforms {
            peptides,
            out,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::peptidoforms::run(stages::peptidoforms::PeptidoformsParams {
                peptides: &peptides,
                out: &out,
                cfg: &cfg.peptidoforms,
                config_hash: &ch,
            })?;
        }
        Cmd::PredictFrag {
            peptidoforms,
            out_precursors,
            out_fragments,
            work_dir,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::predict_frag::run(stages::predict_frag::PredictFragParams {
                peptidoforms: &peptidoforms,
                out_precursors: &out_precursors,
                out_fragments: &out_fragments,
                work_dir: &work_dir,
                cfg: &cfg.predict_frag,
                config_hash: &ch,
            })?;
        }
        Cmd::SearchSeed {
            ms2,
            lib_precursors,
            lib_fragments,
            out,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::search_seed::run(stages::search_seed::SearchSeedParams {
                ms2: &ms2,
                library_precursors: &lib_precursors,
                library_fragments: &lib_fragments,
                out: &out,
                cfg: &cfg.search_seed,
                bucket_size: cfg.extract.bucket_size,
                config_hash: &ch,
            })?;
        }
        Cmd::RtImTrain {
            seed_psms,
            lib_precursors,
            out_windows,
            out_cal,
            config,
        } => {
            let cfg = load_config(&config)?;
            // `finetune_deeplc` is honored only by the `run` / `run-experiment`
            // orchestrators, which invoke the sidecar BEFORE this stage. The
            // standalone stage never fine-tunes, and silently ignoring the flag
            // made 83 production runs read as fine-tuned when they were not.
            if cfg.rt_im_train.finetune_deeplc {
                tracing::warn!(
                    "rt_im_train.finetune_deeplc is set but `mumdia rt-im-train` never \
                     fine-tunes; only `run`/`run-experiment` invoke the DeepLC sidecar. \
                     Pass a pre-fine-tuned --library-precursors table, or use `run`."
                );
            }
            if cfg.rt_im_train.window_holdout_frac > 0.0 && !cfg.rt_im_train.finetune_deeplc {
                tracing::warn!(
                    "rt_im_train.window_holdout_frac is set without finetune_deeplc: the \
                     holdout is honest against this stage's calibration fit, but if the \
                     library iRT came from a fine-tune that saw these anchors, the sizing \
                     is still optimistic"
                );
            }
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::rt_im_train::run(stages::rt_im_train::RtImTrainParams {
                seed_psms: &seed_psms,
                library_precursors: &lib_precursors,
                out_windows: &out_windows,
                out_cal: &out_cal,
                cfg: &cfg.rt_im_train,
                config_hash: &ch,
            })?;
        }
        Cmd::Extract {
            ms2,
            lib_precursors,
            lib_fragments,
            run_windows,
            ms1,
            mass_cal,
            out_psms,
            out_chromatograms,
            restrict_candidates,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::extract::run(stages::extract::ExtractParams {
                ms2: &ms2,
                library_precursors: &lib_precursors,
                library_fragments: &lib_fragments,
                run_windows: &run_windows,
                ms1: ms1.as_deref(),
                mass_cal: mass_cal.as_deref(),
                out_psms: &out_psms,
                out_chrom: &out_chromatograms,
                restrict_candidates: restrict_candidates.as_deref(),
                cfg: &cfg.extract,
                config_hash: &ch,
            })?;
        }
        Cmd::Features {
            psms_extracted,
            chromatograms,
            seed_psms,
            out,
            out_pin,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::features::run(stages::features::FeaturesParams {
                psms: &psms_extracted,
                chromatograms: &chromatograms,
                seed: seed_psms.as_deref(),
                out: &out,
                out_pin: &out_pin,
                cfg: &cfg.features,
                config_hash: &ch,
            })?;
        }
        Cmd::Prescan {
            ms2,
            isolation_windows,
            lib_precursors,
            run_windows,
            out,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::prescan::run(stages::prescan::PrescanParams {
                ms2: &ms2,
                isolation_windows: &isolation_windows,
                library_precursors: &lib_precursors,
                run_windows: &run_windows,
                out: &out,
                cfg: &cfg.prescan,
                config_hash: &ch,
            })?;
        }
        Cmd::Compete {
            features,
            out,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::compete::run(stages::compete::CompeteParams {
                features: &features,
                out: &out,
                cfg: &cfg.compete,
                config_hash: &ch,
            })?;
        }
        Cmd::Audit {
            lib_precursors,
            psms_extracted,
            competed,
            psms_scored,
            out,
            q,
            run_id,
            entrapment_substr,
        } => {
            stages::audit::run(stages::audit::AuditParams {
                library_precursors: &lib_precursors,
                psms: &psms_extracted,
                competed: &competed,
                scored: &psms_scored,
                out: &out,
                q_threshold: q,
                run_id: &run_id,
                entrapment_substr: &entrapment_substr,
            })?;
        }
        Cmd::Rescore {
            competed,
            out,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::rescore::run(stages::rescore::RescoreParams {
                competed: &competed,
                out: &out,
                work_dir: "sidecar_work",
                script_dir: &cfg.predict_frag.sidecar_script_dir,
                cfg: &cfg.rescore,
                config_hash: &ch,
            })?;
        }
        Cmd::Run {
            fasta,
            mzml,
            out_dir,
            lib_precursors,
            lib_fragments,
            config,
            profile,
            max_spectra,
            top_peaks_ms2,
        } => {
            let mut cfg = load_config(&config)?;
            if let Some(pf) = &profile {
                cfg.apply_profile(pf)?;
            }
            stages::run::run(stages::run::RunParams {
                config: &cfg,
                config_path: config.as_deref(),
                fasta: fasta.as_deref(),
                mzml: &mzml,
                out_dir: &out_dir,
                lib_precursors: lib_precursors.as_deref(),
                lib_fragments: lib_fragments.as_deref(),
                max_spectra,
                top_peaks_ms2,
            })?;
        }
        Cmd::RunExperiment {
            fasta,
            mzml,
            run_names,
            out_dir,
            lib_precursors,
            lib_fragments,
            config,
            profile,
            max_spectra,
            top_peaks_ms2,
        } => {
            let mut cfg = load_config(&config)?;
            if let Some(pf) = &profile {
                cfg.apply_profile(pf)?;
            }
            let run_names = if run_names.is_empty() {
                None
            } else {
                Some(run_names.as_slice())
            };
            stages::run_experiment::run(stages::run_experiment::RunExperimentParams {
                config: &cfg,
                config_path: config.as_deref(),
                fasta: fasta.as_deref(),
                mzmls: &mzml,
                run_names,
                out_dir: &out_dir,
                lib_precursors: lib_precursors.as_deref(),
                lib_fragments: lib_fragments.as_deref(),
                max_spectra,
                top_peaks_ms2,
            })?;
        }
        Cmd::Quant {
            psms_scored,
            chromatograms,
            out_peptide,
            out_protein,
            out_fragment,
            out_peak_bounds,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::quant::run(stages::quant::QuantParams {
                psms_scored: &psms_scored,
                chromatograms: &chromatograms,
                out_peptide: &out_peptide,
                out_protein: &out_protein,
                out_fragment: out_fragment.as_deref(),
                out_peak_bounds: out_peak_bounds.as_deref(),
                cfg: &cfg.quant,
                config_hash: &ch,
            })?;
        }
        Cmd::QuantLfq {
            inputs,
            method,
            normalize,
            out,
        } => {
            let by_fragment = method.eq_ignore_ascii_case("directlfq");
            if !by_fragment && !method.eq_ignore_ascii_case("maxlfq") {
                anyhow::bail!("--method must be maxlfq or directlfq (got {method})");
            }
            let norm =
                mumdia_core::config::NormalizeMethod::from_token(&normalize).ok_or_else(|| {
                    anyhow::anyhow!(
                        "--normalize must be median_ratio, median, or none (got {normalize})"
                    )
                })?;
            stages::quant::run_lfq_combine(&inputs, by_fragment, norm, &out)?;
        }
        Cmd::Align {
            seed_psms,
            out,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::align::run(stages::align::AlignParams {
                seeds: &seed_psms,
                out: &out,
                q_train: cfg.rt_im_train.q_train,
                grid_n: 100,
                config_hash: &ch,
            })?;
        }
        Cmd::Mbr {
            psms_scored,
            psms_extracted,
            out,
            out_psms_scored,
            frag,
            config,
        } => {
            let cfg = load_config(&config)?;
            if cfg.mbr.strategy == mumdia_core::config::MbrStrategy::None {
                anyhow::bail!(
                    "mbr.strategy is `none`; set empirical_library / rt_transfer / full to run MBR"
                );
            }
            if psms_extracted.len() < 2 {
                anyhow::bail!(
                    "MBR needs >= 2 runs; got {} psms path(s)",
                    psms_extracted.len()
                );
            }
            let python = cfg.mbr.python.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "mbr.python (sidecar interpreter) is required when mbr.strategy != none"
                )
            })?;
            let script = mumdia::sidecar::resolve_script(
                &cfg.predict_frag.sidecar_script_dir,
                "mbr_worker.py",
            );
            mumdia::sidecar::run_mbr(
                python,
                &script,
                &psms_scored,
                &psms_extracted,
                &out,
                out_psms_scored.as_deref(),
                &frag,
                cfg.mbr.q_anchor,
                cfg.mbr.min_anchor_runs,
                cfg.mbr.q_transfer,
                cfg.mbr.consensus_corr_min,
                cfg.rng_seed,
            )?;
        }
        Cmd::Inspect { artifact } => {
            print!("{}", mumdia_io::inspect(&artifact)?);
        }
        Cmd::Report {
            psms_scored,
            out_dir,
            peptide_quant,
            protein_quant,
            q,
            config,
        } => {
            // Precedence: an explicit --q, then quant.q_threshold from --config, then
            // the historical 0.01. Reading the config is what stops a standalone report
            // silently disagreeing with the `run` that produced the same table.
            let q = match q {
                Some(v) => v,
                None => match &config {
                    Some(_) => load_config(&config)?.quant.q_threshold,
                    None => 0.01,
                },
            };
            std::fs::create_dir_all(&out_dir)?;
            let pep = format!("{out_dir}/peptides.tsv");
            let prot = format!("{out_dir}/proteins.tsv");
            let (n_pep, n_prot) = stages::report::run(stages::report::ReportParams {
                scored: &psms_scored,
                peptide_quant: peptide_quant.as_deref(),
                protein_quant: protein_quant.as_deref(),
                out_peptides: &pep,
                out_proteins: &prot,
                q_threshold: q,
            })?;
            println!(
                "MuMDIA: {n_pep} peptides, {n_prot} protein groups at q <= {q}\n  {pep}\n  {prot}"
            );
        }
        Cmd::Doctor { config, json } => {
            // Deliberately the raw loader: doctor must be able to diagnose a
            // configuration whose interpreters do not resolve, so it cannot go
            // through the resolving loader that would bail first.
            let cfg = load_config_raw(&config)?;
            let report = doctor_report(&cfg, config.as_deref());
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_doctor(&report);
            }
            if !report.ok {
                anyhow::bail!("this configuration cannot run as it stands; see the report above");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    /// The verbosity flags have to keep working after the subcommand as well as
    /// before it, which is what `global = true` buys and what a user will type.
    #[test]
    fn global_flags_parse_on_either_side_of_the_subcommand() {
        let a = Cli::parse_from(["mumdia", "--threads", "8", "doctor"]);
        let b = Cli::parse_from(["mumdia", "doctor", "--threads", "8"]);
        assert_eq!(a.threads, Some(8));
        assert_eq!(b.threads, Some(8));
        assert!(matches!(a.cmd, Cmd::Doctor { .. }));
    }

    #[test]
    fn log_filter_maps_the_verbosity_flags() {
        let f = |args: &[&str]| {
            let mut v = vec!["mumdia"];
            v.extend_from_slice(args);
            v.push("doctor");
            Cli::parse_from(v).log_filter()
        };
        // Nothing given: leave RUST_LOG in charge.
        assert_eq!(f(&[]), None);
        assert_eq!(f(&["-v"]), Some("debug".to_string()));
        assert_eq!(f(&["-vv"]), Some("trace".to_string()));
        assert_eq!(f(&["-vvv"]), Some("trace".to_string()));
        assert_eq!(f(&["-q"]), Some("warn".to_string()));
        // An explicit level wins over the counted flags, and a full RUST_LOG
        // filter passes through unchanged.
        assert_eq!(
            f(&["-vv", "--log-level", "error"]),
            Some("error".to_string())
        );
        assert_eq!(
            f(&["--log-level", "mumdia=debug,extract=trace"]),
            Some("mumdia=debug,extract=trace".to_string())
        );
        // -q and -v contradict each other, so they are rejected rather than
        // silently ordered.
        assert!(Cli::try_parse_from(["mumdia", "-q", "-v", "doctor"]).is_err());
    }

    #[test]
    fn threads_must_be_at_least_one() {
        let err = apply_threads(Some(0)).unwrap_err().to_string();
        assert!(err.contains("--threads"), "{err}");
        // `None` is always fine and must not touch the global pool.
        assert!(apply_threads(None).is_ok());
    }

    use clap::Parser;

    #[test]
    fn conversion_caps_default_to_uncapped() {
        let cli = Cli::try_parse_from([
            "mumdia",
            "convert",
            "--mzml",
            "run.mzML",
            "--out-dir",
            "spectra",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Convert {
                top_peaks_ms2,
                top_peaks_ms1,
                ..
            } => {
                assert_eq!(top_peaks_ms2, 0);
                assert_eq!(top_peaks_ms1, 0);
            }
            _ => panic!("expected convert command"),
        }

        let cli = Cli::try_parse_from([
            "mumdia",
            "run",
            "--fasta",
            "proteome.fasta",
            "--mzml",
            "run.mzML",
            "--out-dir",
            "out",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Run { top_peaks_ms2, .. } => assert_eq!(top_peaks_ms2, 0),
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn explicit_conversion_cap_is_preserved() {
        let cli = Cli::try_parse_from([
            "mumdia",
            "convert",
            "--mzml",
            "run.mzML",
            "--out-dir",
            "spectra",
            "--top-peaks-ms2",
            "300",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Convert { top_peaks_ms2, .. } => assert_eq!(top_peaks_ms2, 300),
            _ => panic!("expected convert command"),
        }
    }
}
