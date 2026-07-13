//! MuMDIA CLI: one binary, one subcommand per stage (PLAN.md Section 3.1, 3.5).
//! Every stage runs standalone on path-addressable inputs.

use anyhow::Result;
use clap::{Parser, Subcommand};
use mumdia::stages;
use mumdia_core::config::Config;

#[derive(Parser)]
#[command(name = "mumdia", version, about = "MuMDIA DIA search engine (Rust MVP)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
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
        /// Keep at most this many MS2 peaks per scan (0 = all).
        #[arg(long, default_value_t = 300)]
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
    /// Native broad DIA seed search over the fragment index -> seed_psms.parquet.
    SearchSeed {
        #[arg(long)]
        ms2: String,
        #[arg(long)]
        library_precursors: String,
        #[arg(long)]
        library_fragments: String,
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
        library_precursors: String,
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
        library_precursors: String,
        #[arg(long)]
        library_fragments: String,
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
        out_chrom: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Compute the minimal feature set -> features.parquet + PIN.
    Features {
        #[arg(long)]
        psms: String,
        #[arg(long)]
        chromatograms: String,
        /// Optional seed_psms for search-engine corroboration features.
        #[arg(long)]
        seed: Option<String>,
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
        #[arg(long)]
        fasta: String,
        #[arg(long)]
        mzml: String,
        #[arg(long)]
        out_dir: String,
        #[arg(long)]
        config: Option<String>,
        /// Named tuning preset applied on top of --config/defaults. "dia" = the
        /// validated DIA preset (Extended features, rolling-window apex, RT prior).
        #[arg(long)]
        profile: Option<String>,
        #[arg(long, default_value_t = 0)]
        max_spectra: usize,
        #[arg(long, default_value_t = 300)]
        top_peaks_ms2: usize,
    },
    /// Cross-run RT alignment (experiment-level) -> alignment.parquet.
    Align {
        /// One seed_psms.parquet per run; the first is the reference.
        #[arg(long, num_args = 1..)]
        seeds: Vec<String>,
        #[arg(long)]
        out: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Print schema, head sample, and row count for any artifact.
    Inspect {
        artifact: String,
    },
}

fn load_config(_path: &Option<String>) -> Result<Config> {
    match _path {
        Some(p) => {
            let s = std::fs::read_to_string(p)?;
            Ok(Config::from_json(&s)?)
        }
        None => Ok(Config::default()),
    }
}

fn main() -> Result<()> {
    mumdia_io::init_logging();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Convert {
            mzml,
            out_dir,
            max_spectra,
            top_peaks_ms2,
            top_peaks_ms1,
        } => {
            let cfg = load_config(&None)?;
            let config_hash = mumdia_io::hash::blake3_str(&cfg.canonical_json());
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
        Cmd::Peptidoforms { peptides, out, config } => {
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
            library_precursors,
            library_fragments,
            out,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::search_seed::run(stages::search_seed::SearchSeedParams {
                ms2: &ms2,
                library_precursors: &library_precursors,
                library_fragments: &library_fragments,
                out: &out,
                cfg: &cfg.search_seed,
                bucket_size: cfg.extract.bucket_size,
                config_hash: &ch,
            })?;
        }
        Cmd::RtImTrain {
            seed_psms,
            library_precursors,
            out_windows,
            out_cal,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::rt_im_train::run(stages::rt_im_train::RtImTrainParams {
                seed_psms: &seed_psms,
                library_precursors: &library_precursors,
                out_windows: &out_windows,
                out_cal: &out_cal,
                cfg: &cfg.rt_im_train,
                config_hash: &ch,
            })?;
        }
        Cmd::Extract {
            ms2,
            library_precursors,
            library_fragments,
            run_windows,
            ms1,
            mass_cal,
            out_psms,
            out_chrom,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::extract::run(stages::extract::ExtractParams {
                ms2: &ms2,
                library_precursors: &library_precursors,
                library_fragments: &library_fragments,
                run_windows: &run_windows,
                ms1: ms1.as_deref(),
                mass_cal: mass_cal.as_deref(),
                out_psms: &out_psms,
                out_chrom: &out_chrom,
                cfg: &cfg.extract,
                config_hash: &ch,
            })?;
        }
        Cmd::Features {
            psms,
            chromatograms,
            seed,
            out,
            out_pin,
            config,
        } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::features::run(stages::features::FeaturesParams {
                psms: &psms,
                chromatograms: &chromatograms,
                seed: seed.as_deref(),
                out: &out,
                out_pin: &out_pin,
                cfg: &cfg.features,
                config_hash: &ch,
            })?;
        }
        Cmd::Compete { features, out, config } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::compete::run(stages::compete::CompeteParams {
                features: &features,
                out: &out,
                cfg: &cfg.compete,
                config_hash: &ch,
            })?;
        }
        Cmd::Rescore { competed, out, config } => {
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
                fasta: &fasta,
                mzml: &mzml,
                out_dir: &out_dir,
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
        Cmd::QuantLfq { inputs, method, normalize, out } => {
            let by_fragment = method.eq_ignore_ascii_case("directlfq");
            if !by_fragment && !method.eq_ignore_ascii_case("maxlfq") {
                anyhow::bail!("--method must be maxlfq or directlfq (got {method})");
            }
            let norm = mumdia_core::config::NormalizeMethod::from_token(&normalize)
                .ok_or_else(|| anyhow::anyhow!("--normalize must be median_ratio, median, or none (got {normalize})"))?;
            stages::quant::run_lfq_combine(&inputs, by_fragment, norm, &out)?;
        }
        Cmd::Align { seeds, out, config } => {
            let cfg = load_config(&config)?;
            let ch = mumdia_io::hash::blake3_str(&cfg.canonical_json());
            stages::align::run(stages::align::AlignParams {
                seeds: &seeds,
                out: &out,
                q_train: cfg.rt_im_train.q_train,
                grid_n: 100,
                config_hash: &ch,
            })?;
        }
        Cmd::Inspect { artifact } => {
            print!("{}", mumdia_io::inspect(&artifact)?);
        }
    }
    Ok(())
}
