//! Vendor-format input: converting a vendor file to mzML before the engine reads
//! it (docs/04_convert.md).
//!
//! # Why conversion is a child process and not a linked reader
//!
//! `mzdata` can read several vendor formats directly, and that was rejected on
//! build grounds rather than capability. Those readers need the vendors' own
//! libraries and, for Thermo and SCIEX, a .NET runtime, while the workspace pins
//! `mzdata` to `default-features = false, features = ["mzml", "miniz_oxide"]`
//! precisely so that building MuMDIA needs no C, C++ or .NET toolchain
//! (`CLAUDE.md`, "Build gotchas: do not fix these back"). Linking a vendor reader
//! imposes that on every build on every platform, including the ones that never see
//! a vendor file.
//!
//! # Two converters, and why
//!
//! ThermoRawFileParser handles Thermo `.raw`. It is Apache-2.0, from CompOmics,
//! and needs no licence acknowledgement, which is why the desktop application can
//! simply install it.
//!
//! Everything else goes through ProteoWizard `msconvert`, which is the only
//! practical converter for Bruker, SCIEX, Agilent and Waters. MuMDIA does not ship
//! or download it: ProteoWizard's own builds bundle vendor libraries under vendor
//! licences that the user accepts when they obtain it, and automating that
//! acceptance is not ours to do. It is located, not installed.
//!
//! # The honest state of each format
//!
//! Thermo is exercised end to end. The rest are wired, documented and unit-tested
//! at the dispatch level, and no Bruker, SCIEX, Agilent or Waters file has been
//! converted by this code. See `docs/04_convert.md`, "Vendor formats", which says
//! the same thing in the place a user looks.
//!
//! Bruker carries a further caveat that is not about this module: MuMDIA's pipeline
//! is 3D and discards ion mobility, so diaPASEF loses the separation that makes it
//! selective. `warn_about_ion_mobility` says so at the point of use.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{bail, Context, Result};
use mumdia_core::config::ConvertConfig;
use tracing::{info, warn};

/// Executable names ThermoRawFileParser ships under.
///
/// The 2.x self-contained builds produce a native `ThermoRawFileParser`
/// (`.exe` on Windows); the 1.4.x line shipped a managed `.exe` that Linux users
/// ran under Mono, which `thermo_command` still handles.
const PARSER_NAMES: &[&str] = &["ThermoRawFileParser.exe", "ThermoRawFileParser"];

/// Executable names msconvert ships under.
const MSCONVERT_NAMES: &[&str] = &["msconvert.exe", "msconvert"];

/// What kind of input a spectra path is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    /// mzML (or anything else `mzdata` reads as-is). No conversion.
    MzMl,
    /// Thermo `.raw`, which is a file.
    Thermo,
    /// Waters `.raw`, which is a DIRECTORY. Same extension as Thermo, different
    /// thing entirely; see `detect`.
    Waters,
    /// Bruker `.d` (timsTOF TDF, or the older BAF).
    Bruker,
    /// Agilent `.d`, which shares Bruker's extension and is also a directory.
    Agilent,
    /// SCIEX `.wiff` / `.wiff2`.
    Sciex,
}

impl Vendor {
    /// Human name, for messages.
    pub fn name(self) -> &'static str {
        match self {
            Vendor::MzMl => "mzML",
            Vendor::Thermo => "Thermo .raw",
            Vendor::Waters => "Waters .raw",
            Vendor::Bruker => "Bruker .d",
            Vendor::Agilent => "Agilent .d",
            Vendor::Sciex => "SCIEX .wiff",
        }
    }

    /// Does this need converting before the engine can read it?
    pub fn needs_conversion(self) -> bool {
        self != Vendor::MzMl
    }
}

/// Classify a spectra path.
///
/// # The `.raw` collision
///
/// Thermo and Waters both use `.raw`, and they are not remotely the same format:
/// Thermo's is a single file, Waters' is a directory of `_FUNC*.DAT` files. So the
/// extension alone cannot decide, and this uses the same discriminator every other
/// tool does, which is whether the path is a file or a directory. A `.raw`
/// directory routed to ThermoRawFileParser would fail with something unhelpful
/// about the file not being readable.
///
/// # The `.d` collision
///
/// Bruker and Agilent both use `.d`, and both are directories. Bruker's contains
/// `analysis.tdf` (timsTOF) or `analysis.baf` (older); Agilent's contains
/// `AcqData/`. Both go to msconvert either way, so this distinction exists only so
/// that the ion-mobility warning fires for Bruker and not for Agilent.
pub fn detect(path: &str) -> Vendor {
    let p = Path::new(path);
    let ext = p
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "raw" => {
            // A directory is Waters. A file, or something that does not exist yet
            // (the error then names the missing path rather than the format), is
            // Thermo, which is overwhelmingly the common case.
            if p.is_dir() {
                Vendor::Waters
            } else {
                Vendor::Thermo
            }
        }
        "d" => {
            if p.join("analysis.tdf").exists() || p.join("analysis.baf").exists() {
                Vendor::Bruker
            } else if p.join("AcqData").exists() {
                Vendor::Agilent
            } else {
                // Unrecognised contents, or a path that does not exist. Bruker is
                // the far more common `.d` in DIA proteomics, and the warning it
                // carries is the one worth erring towards.
                Vendor::Bruker
            }
        }
        "wiff" | "wiff2" => Vendor::Sciex,
        _ => Vendor::MzMl,
    }
}

/// Is this path a Thermo `.raw` file?
///
/// Kept as its own function because callers outside this module ask exactly this
/// question, and because the desktop application mirrors it.
pub fn is_thermo_raw(path: &str) -> bool {
    detect(path) == Vendor::Thermo
}

/// Locate ThermoRawFileParser.
///
/// An explicit path is taken as given, and its absence is an error rather than a
/// fallback to whatever else is on the machine. That is deliberate: vendor
/// conversion is not reproducible across converters or even across converter
/// versions, so silently using a different one than the configuration named would
/// change the spectra a search sees without saying so.
///
/// `"auto"` (or empty) searches `MUMDIA_THERMO_PARSER`, then beside the engine
/// binary, then `PATH`. The current working directory is deliberately NOT searched,
/// for the reason `python::resolve_script_dir` documents at length: an untrusted
/// input directory containing a file with the right name would otherwise be
/// executed.
pub fn locate_parser(configured: &str) -> Result<PathBuf> {
    // The literal read, rather than passing the name into the shared helper: the
    // documentation generator scans for `std::env::var_os(` with a literal argument
    // and cannot follow a `&str` parameter, so an indirected read disappears from
    // the generated environment-variable table.
    let from_env = std::env::var_os("MUMDIA_THERMO_PARSER");
    locate(
        configured,
        "convert.thermo_raw_parser",
        "MUMDIA_THERMO_PARSER",
        from_env,
        PARSER_NAMES,
        &["ThermoRawFileParser"],
        Vec::new(),
        "no ThermoRawFileParser was found, so a Thermo .raw file cannot be read. \
         Install it (https://github.com/compomics/ThermoRawFileParser, Apache-2.0), \
         then set convert.thermo_raw_parser or MUMDIA_THERMO_PARSER to its \
         executable. Alternatively convert to mzML yourself and pass that.",
    )
}

/// Locate ProteoWizard `msconvert`.
///
/// Same rules as `locate_parser`, plus the Windows install layout: ProteoWizard
/// installs into a version-stamped directory under Program Files, so those are
/// searched and the highest version wins.
pub fn locate_msconvert(configured: &str) -> Result<PathBuf> {
    let from_env = std::env::var_os("MUMDIA_MSCONVERT");
    let mut extra: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        // Two plain `let` reads rather than a loop over the names, and rather than
        // `if let Some(v) = env::var_os("...")`. `ci/gen_config_reference.py`
        // resolves a read only from a literal argument in this position, so both a
        // loop variable and an `if let` binding are reported as unresolved reads and
        // these two roots go undocumented.
        let pf = std::env::var_os("ProgramFiles");
        let pf_x86 = std::env::var_os("ProgramFiles(x86)");
        let program_files: Vec<PathBuf> = [pf, pf_x86]
            .into_iter()
            .flatten()
            .map(PathBuf::from)
            .collect();
        for base in program_files {
            let root = base.join("ProteoWizard");
            let Ok(entries) = std::fs::read_dir(&root) else {
                continue;
            };
            let mut subs: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            // Sorted so the choice between installed versions is deterministic, and
            // reversed so the newest wins. Version strings sort lexicographically
            // here because ProteoWizard stamps them `3.0.<build>.<hash>` with a
            // fixed-width build number.
            subs.sort();
            extra.extend(subs.into_iter().rev());
            extra.push(root);
        }
    }
    locate(
        configured,
        "convert.msconvert",
        "MUMDIA_MSCONVERT",
        from_env,
        MSCONVERT_NAMES,
        &[],
        extra,
        "no msconvert was found, so this vendor format cannot be read. Install \
         ProteoWizard (https://proteowizard.sourceforge.io/), then set \
         convert.msconvert or MUMDIA_MSCONVERT to msconvert. Alternatively convert \
         to mzML yourself and pass that. Note that ProteoWizard's vendor readers \
         carry the instrument vendors' own licence terms, which you accept when you \
         obtain it; MuMDIA neither ships nor downloads it.",
    )
}

/// The shared search. See `locate_parser` for the reasoning.
///
/// `from_env` is the already-read environment override; `extra` roots are appended
/// after the engine-relative ones and before `PATH`.
#[allow(clippy::too_many_arguments)]
fn locate(
    configured: &str,
    field: &str,
    env_var: &str,
    from_env: Option<std::ffi::OsString>,
    names: &[&str],
    subdirs: &[&str],
    extra: Vec<PathBuf>,
    not_found: &str,
) -> Result<PathBuf> {
    let configured = configured.trim();
    if !configured.is_empty() && configured != "auto" {
        let p = PathBuf::from(configured);
        if p.is_file() {
            return Ok(p);
        }
        bail!(
            "{field} is set to {configured}, which is not a file. Correct the path, \
             or set it to \"auto\" to search for a converter."
        );
    }

    if let Some(v) = from_env {
        let p = PathBuf::from(&v);
        if p.is_file() {
            return Ok(p);
        }
        warn!(
            path = %p.display(),
            "{env_var} is set but does not point at a file; continuing the search"
        );
    }

    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
            for sub in subdirs {
                roots.push(dir.join(sub));
                roots.push(dir.join("binaries").join(sub));
            }
        }
    }
    roots.extend(extra);
    if let Some(path) = std::env::var_os("PATH") {
        roots.extend(std::env::split_paths(&path));
    }
    for dir in roots {
        for name in names {
            let cand = dir.join(name);
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    bail!("{not_found}")
}

/// Was a converter path named explicitly, rather than left to the search?
///
/// `"auto"` and empty mean "search"; anything else is a deliberate choice whose
/// failure must not be papered over with a different program.
fn is_explicit(configured: &str) -> bool {
    let c = configured.trim();
    !c.is_empty() && c != "auto"
}

/// Build the command that runs ThermoRawFileParser.
///
/// A managed `.exe` on a Unix host is the 1.4.x distribution, which needs Mono. A
/// self-contained 2.x build is executed directly on both platforms.
fn thermo_command(parser: &Path) -> std::process::Command {
    let managed_exe_on_unix = cfg!(unix)
        && parser
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("exe"));
    if managed_exe_on_unix {
        let mut cmd = std::process::Command::new("mono");
        cmd.arg(parser);
        cmd
    } else {
        std::process::Command::new(parser)
    }
}

/// Is `candidate` newer than `source`?
///
/// Both times must be readable to answer yes. An unreadable timestamp means the
/// question cannot be settled, and the safe answer is to convert again rather than
/// to reuse a file that may predate the input.
///
/// For a directory input (Bruker/Agilent `.d`, Waters `.raw`) the directory's own
/// mtime is not reliable -- on some filesystems it does not change when a contained
/// file is rewritten -- so the newest mtime anywhere inside it is used.
fn is_newer_than(candidate: &Path, source: &Path) -> bool {
    let Ok(c) = std::fs::metadata(candidate).and_then(|m| m.modified()) else {
        return false;
    };
    let Some(s) = newest_mtime(source) else {
        return false;
    };
    c >= s
}

fn newest_mtime(path: &Path) -> Option<std::time::SystemTime> {
    let md = std::fs::metadata(path).ok()?;
    let mut newest = md.modified().ok()?;
    if md.is_dir() {
        // One level is enough for every layout here: the acquisition files sit
        // directly in the `.d` / `.raw` directory, and a full recursive walk of a
        // multi-gigabyte directory to answer a caching question is not worth it.
        if let Ok(entries) = std::fs::read_dir(path) {
            for e in entries.flatten() {
                if let Ok(t) = e.metadata().and_then(|m| m.modified()) {
                    if t > newest {
                        newest = t;
                    }
                }
            }
        }
    }
    Some(newest)
}

/// Can this directory be written to? Answered by trying, because a permission bit
/// is not the whole story on a network share.
fn is_writable(dir: &Path) -> bool {
    let probe = dir.join(".mumdia-write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Say, once and loudly, that ion mobility is discarded.
///
/// Not an error. Collapsing the mobility dimension costs sensitivity rather than
/// FDR validity: targets and decoys see the same added interference, so the
/// threshold stays calibrated while fewer things pass it. A user with non-PASEF
/// Bruker DIA is also perfectly well served. But a diaPASEF user who is not told
/// this will read a low identification count as a MuMDIA result rather than as the
/// cost of throwing away the separation their acquisition exists to produce.
fn warn_about_ion_mobility(vendor: Vendor) {
    if vendor != Vendor::Bruker {
        return;
    }
    warn!(
        "convert: MuMDIA's pipeline is 3D and discards ion mobility. For diaPASEF \
         this removes the mobility separation that makes the acquisition selective, \
         so expect substantially more interference and fewer identifications than a \
         4D engine on the same file. The q values stay calibrated (targets and \
         decoys see the same interference); the sensitivity does not. See README, \
         \"No ion mobility\"."
    );
}

/// The msconvert arguments for a vendor, before the user's own additions.
///
/// Vendor peak picking is requested where it exists and is better than the
/// local-maxima fallback in `stages::convert`. It is NOT requested for Bruker: TDF
/// data is already centroided and msconvert rejects the filter there. Bruker gets
/// `--combineIonMobilitySpectra` instead, which is what turns a mobility-resolved
/// frame into the 3D spectra this pipeline reads; without it the output is one
/// spectrum per mobility scan, which is both enormous and not what any downstream
/// stage expects.
fn msconvert_vendor_args(vendor: Vendor) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "--mzML".into(),
        "--64".into(),
        "--zlib".into(),
        // Indexed output, matching what the engine has always read.
        "--simAsSpectra".into(),
    ];
    // `--simAsSpectra` is harmless where SIM is absent and avoids silently dropping
    // SIM scans where it is not.
    match vendor {
        Vendor::Bruker => a.push("--combineIonMobilitySpectra".into()),
        Vendor::MzMl | Vendor::Thermo | Vendor::Waters | Vendor::Agilent | Vendor::Sciex => {
            a.push("--filter".into());
            a.push("peakPicking vendor msLevel=1-".into());
        }
    }
    a
}

/// A short, stable discriminator for an input path.
///
/// Used to keep two inputs that share a file stem from colliding on one converted
/// mzML. Derived from the absolute path so it is stable across runs; the hash is
/// truncated because it only has to separate the handful of inputs in one experiment,
/// not resist collision attacks.
fn path_discriminator(input: &Path) -> String {
    use std::hash::{Hash, Hasher};
    let abs = input
        .canonicalize()
        .unwrap_or_else(|_| input.to_path_buf())
        .to_string_lossy()
        .to_lowercase();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    abs.hash(&mut h);
    format!("{:08x}", h.finish() as u32)
}

/// Is there another vendor input in `dir` that would produce the same mzML name?
///
/// `sample.raw` and `sample.d` in one folder both reduce to `sample`, and the second
/// conversion would silently adopt the first's output through `reuse_converted`.
fn stem_is_ambiguous(dir: &Path, stem: &str, this: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.as_path() != this)
        .filter(|p| detect(&p.to_string_lossy()).needs_conversion())
        .any(|p| p.file_stem().is_some_and(|f| f.to_string_lossy() == stem))
}

/// Return an mzML path for `input`, converting a vendor file first if needed.
///
/// Anything already readable is returned unchanged, so every caller can route its
/// spectra path through this without caring what format it is.
///
/// The converted file goes beside the input when that directory is writable, which
/// makes it reusable across runs and findable afterwards, and into `fallback_dir`
/// otherwise (a read-only share being the ordinary case). With
/// `convert.reuse_converted` an mzML already sitting beside the input and newer
/// than it is used as-is.
/// Name the missing `.wiff.scan` when a SCIEX conversion fails without one.
///
/// SCIEX keeps the spectra in a `.wiff.scan` companion beside the `.wiff`; the `.wiff`
/// alone holds metadata and chromatograms, so a `.wiff` downloaded on its own converts to
/// nothing. msconvert reports that as `[ExperimentImpl::initializeBPC()] Error processing
/// ... Could not open data stream. Is a required 'scan' file missing?` (measured
/// 2026-09-06 on PRIDE-archived `.wiff` files that had no companion), which does not name
/// the file. `.wiff2` carries its own data and gets no hint.
fn sciex_scan_hint(src: &Path) -> Option<String> {
    let name = src.file_name()?.to_string_lossy().into_owned();
    if !name.to_ascii_lowercase().ends_with(".wiff") {
        return None;
    }
    let scan = src.with_file_name(format!("{name}.scan"));
    if scan.exists() {
        return None;
    }
    Some(format!(
        "\n\nNo {} beside the input. SCIEX stores the spectra in that companion file and \
         msconvert cannot convert a .wiff without it; copy both files together.",
        scan.display()
    ))
}

/// The name a conversion writes to before the rename to `out_name`.
///
/// The marker sits in the stem and the `.mzML` extension stays last, because both
/// converters treat the output path's extension as theirs to fix up. ThermoRawFileParser
/// appends `.mzML` to an output path that does not end in it: given `-b x.mzML.partial` it
/// wrote `x.mzML.partial.mzML`, the success check found nothing at `x.mzML.partial`, and a
/// 6:48 conversion of a 3.7 GB Astral run was reported as "exited successfully but wrote no
/// file" and discarded (doxy, 2026-09-06). msconvert's `--outfile` has the same habit. With
/// `x.partial.mzML` there is nothing for either to fix up.
fn partial_name(out_name: &str) -> String {
    let stem = out_name.strip_suffix(".mzML").unwrap_or(out_name);
    format!("{stem}.partial.mzML")
}

pub fn ensure_mzml(
    input: &str,
    cfg: &ConvertConfig,
    fallback_dir: Option<&Path>,
) -> Result<String> {
    let vendor = detect(input);
    if !vendor.needs_conversion() {
        return Ok(input.to_string());
    }
    let src = Path::new(input);
    // `exists`, not `is_file`: Bruker/Agilent `.d` and Waters `.raw` are directories.
    if !src.exists() {
        bail!("{input} does not exist");
    }
    let stem = src
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "converted".to_string());
    let src_dir = src.parent().unwrap_or(Path::new("."));

    // The converted name. Plain `<stem>.mzML` is the useful default -- it is what
    // msconvert would have produced and what a user expects to find beside their data
    // -- but it is only safe while no other vendor input in the same directory shares
    // the stem. When one does, both would resolve to the same file and the second
    // would silently adopt the first's conversion.
    let mzml_name = if stem_is_ambiguous(src_dir, &stem, src) {
        let d = path_discriminator(src);
        warn!(
            stem = %stem,
            dir = %src_dir.display(),
            "convert: another vendor file in this directory has the same name, so the \
             converted mzML is disambiguated rather than shared"
        );
        format!("{stem}.{d}.mzML")
    } else {
        format!("{stem}.mzML")
    };

    // A usable conversion may already be there from a previous run.
    let beside = src_dir.join(&mzml_name);
    if cfg.reuse_converted && beside.is_file() && is_newer_than(&beside, src) {
        info!(
            mzml = %beside.display(),
            vendor = vendor.name(),
            "convert: reusing the existing mzML beside the input (convert.reuse_converted)"
        );
        return Ok(beside.to_string_lossy().into_owned());
    }

    warn_about_ion_mobility(vendor);

    // Thermo goes to ThermoRawFileParser when one is available, because it needs no
    // vendor licence and the desktop application can install it. msconvert is the
    // fallback there and the only option everywhere else.
    // Thermo prefers ThermoRawFileParser, because it needs no vendor licence and the
    // desktop application can install it; msconvert is a legitimate alternative
    // there and the only option everywhere else.
    //
    // The fallback applies ONLY when the parser was left at `"auto"` and nothing was
    // found. An explicitly configured path that is wrong is an error, exactly as
    // `locate_parser` documents: falling through to a different converter would
    // convert with a program the configuration did not name, and vendor conversion
    // is not reproducible across converters.
    let (exe, is_thermo) = if vendor == Vendor::Thermo {
        match locate_parser(&cfg.thermo_raw_parser) {
            Ok(p) => (p, true),
            Err(e) if is_explicit(&cfg.thermo_raw_parser) => return Err(e),
            Err(thermo_err) => match locate_msconvert(&cfg.msconvert) {
                Ok(p) => (p, false),
                // Name both ways out. The Apache-2.0 converter is the easier one to
                // obtain and the one the desktop application installs.
                Err(e) => bail!(
                    "{thermo_err}

msconvert was not usable either: {e}"
                ),
            },
        }
    } else {
        (locate_msconvert(&cfg.msconvert)?, false)
    };

    let out_dir = if is_writable(src_dir) {
        src_dir.to_path_buf()
    } else if let Some(d) = fallback_dir {
        std::fs::create_dir_all(d)
            .with_context(|| format!("creating {} for the converted mzML", d.display()))?;
        warn!(
            src_dir = %src_dir.display(),
            out_dir = %d.display(),
            "convert: the input directory is not writable, so the converted mzML goes to \
             the output directory and will be converted again on the next run"
        );
        d.to_path_buf()
    } else {
        bail!(
            "{} is not writable and no output directory was given, so there is nowhere \
             to put the converted mzML",
            src_dir.display()
        );
    };
    // Writing into the fallback (output) directory pools inputs from EVERY source
    // directory into one place, so the stem alone is never enough there: two runs at
    // /a/sample.raw and /b/sample.raw would land on one file, and an experiment would
    // pool two copies of one run. The discriminator is unconditional in that case.
    let out_name = if out_dir == src_dir {
        mzml_name.clone()
    } else {
        format!("{stem}.{}.mzML", path_discriminator(src))
    };
    let out = out_dir.join(&out_name);
    // Convert to a temporary name and rename only on success.
    //
    // Writing straight to `out` meant a killed run, a power loss or a converter crash
    // left a TRUNCATED mzML at the final path. It is newer than its source, so
    // `reuse_converted` accepted it on every later run: a partial acquisition searched
    // silently, for ever, presenting as unexplained low identification counts with
    // nothing in the interface to reveal it. Neither failure path removed it.
    let tmp = out_dir.join(partial_name(&out_name));
    let _ = std::fs::remove_file(&tmp);

    let args: Vec<String> = if is_thermo {
        // `-f 2` is indexed mzML, which is what msconvert produces by default and so
        // what the engine has always read. `-m 2` suppresses the metadata sidecar,
        // which nothing here consumes. Peak picking is ON by default and is left on:
        // the native Thermo centroiding is better than the local-maxima fallback in
        // `stages::convert`, which then sees centroided input and does nothing.
        vec![
            "-i".into(),
            src.to_string_lossy().into_owned(),
            "-b".into(),
            tmp.to_string_lossy().into_owned(),
            "-f".into(),
            "2".into(),
            "-m".into(),
            "2".into(),
        ]
    } else {
        let mut a = vec![src.to_string_lossy().into_owned()];
        a.extend(msconvert_vendor_args(vendor));
        a.extend(cfg.msconvert_args.iter().cloned());
        // `--outfile` rather than relying on msconvert's naming, so the reuse check
        // above looks for the file this actually wrote.
        a.push("-o".into());
        a.push(out_dir.to_string_lossy().into_owned());
        a.push("--outfile".into());
        a.push(partial_name(&out_name));
        a
    };

    info!(
        converter = %exe.display(),
        vendor = vendor.name(),
        input = %src.display(),
        mzml = %out.display(),
        "convert: converting to mzML (this takes minutes for a large file)"
    );

    let started = std::time::Instant::now();
    let mut cmd = if is_thermo {
        thermo_command(&exe)
    } else {
        std::process::Command::new(&exe)
    };
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .with_context(|| format!("starting {}", exe.display()))?;

    // Both streams on their own threads, and NOT sequentially.
    //
    // Draining stdout to EOF and only then reading stderr deadlocks: a converter that
    // fills the stderr pipe (~64 KB) while this thread is blocked on stdout stops
    // being scheduled, so it never closes stdout, so this never moves on to stderr.
    // Both msconvert and ThermoRawFileParser write progress to both streams on a
    // multi-gigabyte file, which is exactly when it would hang -- and the hang would
    // look like a slow conversion, the one thing the streaming exists to rule out.
    //
    // `diann.rs::run_step` already did this correctly; this function did not, which is
    // the argument for the two sharing an implementation rather than each growing one.
    let mut readers = Vec::new();
    for stream in [
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        readers.push(std::thread::spawn(move || {
            let mut tail: Vec<String> = Vec::new();
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                info!(target: "vendor_converter", "{line}");
                tail.push(line);
                if tail.len() > 40 {
                    tail.remove(0);
                }
            }
            tail
        }));
    }
    let mut tail: Vec<String> = Vec::new();
    for r in readers {
        if let Ok(part) = r.join() {
            tail.extend(part);
        }
    }
    if tail.len() > 80 {
        let drop = tail.len() - 80;
        tail.drain(0..drop);
    }
    let status = child
        .wait()
        .with_context(|| format!("waiting for {}", exe.display()))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        let hint = if vendor == Vendor::Sciex {
            sciex_scan_hint(src).unwrap_or_default()
        } else {
            String::new()
        };
        bail!(
            "{} failed ({}). Last output:\n{}{}",
            exe.file_name().unwrap_or_default().to_string_lossy(),
            status
                .code()
                .map(|c| format!("exit {c}"))
                .unwrap_or_else(|| "killed by a signal".into()),
            tail.join("\n"),
            hint
        );
    }
    // A zero exit with no file is a real outcome, not a hypothetical: both
    // converters report some input problems on stdout and still exit cleanly.
    if !tmp.is_file() {
        bail!(
            "the converter exited successfully but wrote no {}. Last output:\n{}",
            tmp.display(),
            tail.join("\n")
        );
    }
    // Rename last: until this succeeds there is no file at `out`, so a crash
    // anywhere above leaves nothing for `reuse_converted` to mistake for a
    // finished conversion.
    std::fs::rename(&tmp, &out).with_context(|| {
        format!(
            "renaming {} to {} after conversion",
            tmp.display(),
            out.display()
        )
    })?;
    info!(
        mzml = %out.display(),
        elapsed_s = started.elapsed().as_secs(),
        "convert: converted"
    );
    Ok(out.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia-raw-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_wiff_without_its_scan_companion_gets_a_named_hint() {
        // msconvert's own message for this is "Is a required 'scan' file missing?", which
        // names no file; the hint names the one that is missing, and only for `.wiff`.
        let d = tmp("wiff_scan_hint");
        let wiff = d.join("run.wiff");
        std::fs::write(&wiff, b"").unwrap();
        let hint = sciex_scan_hint(&wiff).expect("a lone .wiff must produce a hint");
        assert!(hint.contains("run.wiff.scan"), "{hint}");
        std::fs::write(d.join("run.wiff.scan"), b"").unwrap();
        assert!(
            sciex_scan_hint(&wiff).is_none(),
            "companion present: no hint"
        );
        assert!(
            sciex_scan_hint(&d.join("run.wiff2")).is_none(),
            ".wiff2 needs none"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_partial_name_keeps_the_mzml_extension() {
        // The regression this guards: `x.mzML.partial` made ThermoRawFileParser write
        // `x.mzML.partial.mzML`, and the conversion was thrown away as "wrote no file".
        assert_eq!(partial_name("run.mzML"), "run.partial.mzML");
        assert_eq!(
            partial_name("run.1a2b3c4d.mzML"),
            "run.1a2b3c4d.partial.mzML"
        );
        assert!(partial_name("odd").ends_with(".mzML"));
        assert_ne!(partial_name("run.mzML"), "run.mzML");
    }

    #[test]
    fn mzml_and_unknown_extensions_need_no_conversion() {
        for p in ["a.mzML", "a.mzml", "a.mzXML", "noextension", "a.txt"] {
            assert_eq!(detect(p), Vendor::MzMl, "{p}");
            assert!(!detect(p).needs_conversion(), "{p}");
        }
    }

    #[test]
    fn a_raw_file_is_thermo_and_a_raw_directory_is_waters() {
        // The collision that extension matching alone gets wrong. A Waters `.raw`
        // directory handed to ThermoRawFileParser fails with something unhelpful
        // about an unreadable file.
        let d = tmp("raw-collision");
        let file = d.join("thermo.raw");
        std::fs::write(&file, b"x").unwrap();
        let dir = d.join("waters.raw");
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(detect(file.to_str().unwrap()), Vendor::Thermo);
        assert_eq!(detect(dir.to_str().unwrap()), Vendor::Waters);
        // A path that does not exist yet is treated as Thermo, which is both the
        // common case and the one whose error message names the missing file.
        assert_eq!(detect("/nowhere/x.raw"), Vendor::Thermo);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_d_directory_is_bruker_or_agilent_by_its_contents() {
        // Both are directories with the same extension. The distinction exists only
        // so the ion-mobility warning fires for Bruker and not for Agilent.
        let d = tmp("d-collision");

        let tdf = d.join("tims.d");
        std::fs::create_dir_all(&tdf).unwrap();
        std::fs::write(tdf.join("analysis.tdf"), b"x").unwrap();
        assert_eq!(detect(tdf.to_str().unwrap()), Vendor::Bruker);

        let baf = d.join("old.d");
        std::fs::create_dir_all(&baf).unwrap();
        std::fs::write(baf.join("analysis.baf"), b"x").unwrap();
        assert_eq!(detect(baf.to_str().unwrap()), Vendor::Bruker);

        let agilent = d.join("agilent.d");
        std::fs::create_dir_all(agilent.join("AcqData")).unwrap();
        assert_eq!(detect(agilent.to_str().unwrap()), Vendor::Agilent);

        // Unrecognised contents fall to Bruker, which is the commoner `.d` here and
        // the one that carries the warning worth erring towards.
        let empty = d.join("empty.d");
        std::fs::create_dir_all(&empty).unwrap();
        assert_eq!(detect(empty.to_str().unwrap()), Vendor::Bruker);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn sciex_covers_both_wiff_generations() {
        assert_eq!(detect("a.wiff"), Vendor::Sciex);
        assert_eq!(detect("a.wiff2"), Vendor::Sciex);
        assert_eq!(detect("a.WIFF"), Vendor::Sciex);
    }

    #[test]
    fn a_non_vendor_path_is_returned_untouched_and_needs_no_converter() {
        // The point of routing every caller through `ensure_mzml`: an mzML user must
        // not need any converter to exist, so this path cannot consult one.
        let cfg = ConvertConfig {
            thermo_raw_parser: "/definitely/not/here".to_string(),
            msconvert: "/definitely/not/here".to_string(),
            msconvert_args: Vec::new(),
            reuse_converted: true,
        };
        assert_eq!(ensure_mzml("run.mzML", &cfg, None).unwrap(), "run.mzML");
    }

    #[test]
    fn an_explicit_converter_path_that_is_wrong_is_an_error_not_a_fallback() {
        for (err, field) in [
            (
                locate_parser("/definitely/not/here/ThermoRawFileParser").unwrap_err(),
                "convert.thermo_raw_parser",
            ),
            (
                locate_msconvert("/definitely/not/here/msconvert").unwrap_err(),
                "convert.msconvert",
            ),
        ] {
            let msg = err.to_string();
            assert!(msg.contains("not a file"), "{msg}");
            assert!(
                msg.contains(field),
                "the message must name the field: {msg}"
            );
            assert!(msg.contains("auto"), "and how to search: {msg}");
        }
    }

    #[test]
    fn the_msconvert_error_states_who_owns_the_vendor_licences() {
        // MuMDIA does not ship or download ProteoWizard, and the reason is that its
        // vendor readers carry the instrument vendors' terms. Someone hitting this
        // message is exactly the person who needs to know that.
        let err = locate_msconvert("auto");
        if let Err(e) = err {
            let m = e.to_string();
            assert!(m.contains("vendor"), "{m}");
            assert!(m.contains("neither ships nor downloads"), "{m}");
        }
    }

    #[test]
    fn the_auto_search_never_looks_in_the_working_directory() {
        // Same hazard `python::resolve_script_dir` documents: an untrusted input
        // directory holding a file with this name would otherwise be executed.
        let dir = tmp("cwd-probe");
        for name in [
            "ThermoRawFileParser",
            "ThermoRawFileParser.exe",
            "msconvert",
            "msconvert.exe",
        ] {
            std::fs::write(dir.join(name), b"not a real converter").unwrap();
        }
        let previous = std::env::current_dir().ok();
        // `set_current_dir` is process-wide and the suite is threaded, so this only
        // asserts the negative: whatever is found, it must not be from here.
        std::env::set_current_dir(&dir).unwrap();
        let found = [locate_parser("auto"), locate_msconvert("auto")];
        if let Some(p) = previous {
            let _ = std::env::set_current_dir(p);
        }
        for f in found.into_iter().flatten() {
            let parent = f.parent().and_then(|p| p.canonicalize().ok());
            assert_ne!(
                parent,
                dir.canonicalize().ok(),
                "the working directory must never win the converter search"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_neighbouring_mzml_is_not_reused() {
        // Reuse is keyed on the mzML being NEWER than the input. An mzML that
        // predates it is either from a different file of the same name or from
        // before the input was re-acquired, and using it would search the wrong data.
        let d = tmp("staleness");
        let mzml = d.join("x.mzML");
        let raw = d.join("x.raw");
        std::fs::write(&mzml, b"old").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&raw, b"new").unwrap();

        assert!(!is_newer_than(&mzml, &raw));
        assert!(is_newer_than(&raw, &mzml));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn staleness_of_a_directory_input_looks_inside_it() {
        // A `.d` directory's own mtime does not necessarily change when a contained
        // acquisition file is rewritten, so a stale mzML would have looked reusable.
        let d = tmp("dir-staleness");
        let acq = d.join("run.d");
        std::fs::create_dir_all(&acq).unwrap();
        std::fs::write(acq.join("analysis.tdf"), b"v1").unwrap();
        let mzml = d.join("run.mzML");
        std::fs::write(&mzml, b"converted").unwrap();
        assert!(is_newer_than(&mzml, &acq), "fresh conversion is reusable");

        std::thread::sleep(std::time::Duration::from_millis(20));
        // Re-acquired: a file inside changes, the directory mtime may not.
        std::fs::write(acq.join("analysis.tdf"), b"v2").unwrap();
        assert!(
            !is_newer_than(&mzml, &acq),
            "a rewritten file inside the .d must invalidate the cached mzML"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_writable_directory_is_detected_by_writing_to_it() {
        let d = tmp("writable");
        assert!(is_writable(&d));
        // And the probe file must not survive the check.
        assert!(!d.join(".mumdia-write-probe").exists());
        assert!(!is_writable(&d.join("does-not-exist")));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn bruker_gets_mobility_combining_and_no_vendor_peak_picking() {
        // Both halves matter. Without `--combineIonMobilitySpectra` the output is one
        // spectrum per mobility scan, which no downstream stage expects. And
        // msconvert rejects `peakPicking vendor` on TDF, which is already centroided.
        let b = msconvert_vendor_args(Vendor::Bruker).join(" ");
        assert!(b.contains("--combineIonMobilitySpectra"), "{b}");
        assert!(!b.contains("peakPicking"), "{b}");

        for v in [Vendor::Sciex, Vendor::Waters, Vendor::Agilent] {
            let a = msconvert_vendor_args(v).join(" ");
            assert!(a.contains("peakPicking vendor"), "{}: {a}", v.name());
            assert!(!a.contains("combineIonMobility"), "{}: {a}", v.name());
        }
    }

    #[test]
    fn every_vendor_that_needs_conversion_has_a_name_and_a_route() {
        // A new variant must not be able to reach `ensure_mzml` without arguments.
        for v in [
            Vendor::Thermo,
            Vendor::Waters,
            Vendor::Bruker,
            Vendor::Agilent,
            Vendor::Sciex,
        ] {
            assert!(v.needs_conversion(), "{}", v.name());
            assert!(!v.name().is_empty());
            assert!(
                !msconvert_vendor_args(v).is_empty(),
                "{} has no msconvert arguments",
                v.name()
            );
        }
        assert!(!Vendor::MzMl.needs_conversion());
    }

    #[test]
    fn an_explicit_thermo_parser_that_is_wrong_does_not_fall_back_to_msconvert() {
        // The contract `locate_parser` documents, asserted at the dispatch level:
        // converting with a program the configuration did not name would change the
        // spectra a search sees, because vendor conversion is not reproducible
        // across converters. Only `"auto"` may fall through.
        assert!(is_explicit("/some/path"));
        assert!(!is_explicit("auto"));
        assert!(!is_explicit(""));
        assert!(!is_explicit("  "));

        let d = tmp("explicit-no-fallback");
        let raw = d.join("x.raw");
        std::fs::write(&raw, b"x").unwrap();
        let cfg = ConvertConfig {
            thermo_raw_parser: "/definitely/not/here/ThermoRawFileParser".to_string(),
            // Left searchable on purpose: even a usable msconvert must not be
            // substituted for an explicitly named parser.
            msconvert: "auto".to_string(),
            msconvert_args: Vec::new(),
            reuse_converted: false,
        };
        let err = ensure_mzml(raw.to_str().unwrap(), &cfg, Some(&d)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("convert.thermo_raw_parser"), "{msg}");
        assert!(
            !msg.contains("msconvert was not usable"),
            "an explicit parser path must fail on its own terms, not report a fallback: {msg}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn two_inputs_sharing_a_stem_do_not_collapse_onto_one_mzml() {
        // The corrupting case. Under `run-experiment` every input is mapped through
        // `ensure_mzml` with the output directory as the fallback, so two runs at
        // /a/sample.raw and /b/sample.raw on a read-only share both resolved to
        // out_dir/sample.mzML: the second conversion overwrote the first and the
        // returned list held one path twice, so the experiment pooled two copies of a
        // single run under two run names.
        let d = tmp("stem-collision");
        let a = d.join("plateA");
        let b = d.join("plateB");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let ra = a.join("sample.raw");
        let rb = b.join("sample.raw");
        std::fs::write(&ra, b"a").unwrap();
        std::fs::write(&rb, b"b").unwrap();

        assert_ne!(
            path_discriminator(&ra),
            path_discriminator(&rb),
            "two different inputs must not share a discriminator"
        );
        // Stable across calls, or the cache would miss every run.
        assert_eq!(path_discriminator(&ra), path_discriminator(&ra));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_vendor_file_beside_a_same_named_directory_input_is_disambiguated() {
        // `sample.raw` and `sample.d` in one folder both reduce to `sample`, so the
        // second would have adopted the first's conversion through `reuse_converted`.
        let d = tmp("stem-ambiguous");
        let raw = d.join("sample.raw");
        std::fs::write(&raw, b"x").unwrap();
        let dotd = d.join("sample.d");
        std::fs::create_dir_all(&dotd).unwrap();
        std::fs::write(dotd.join("analysis.tdf"), b"x").unwrap();

        assert!(
            stem_is_ambiguous(&d, "sample", &raw),
            "a same-stemmed sibling vendor input must be detected"
        );
        assert!(stem_is_ambiguous(&d, "sample", &dotd));

        // An mzML sibling is not a vendor input and must not trigger it: that is the
        // ordinary reuse case, not a collision.
        let e = tmp("stem-unambiguous");
        let only = e.join("sample.raw");
        std::fs::write(&only, b"x").unwrap();
        std::fs::write(e.join("sample.mzML"), b"x").unwrap();
        assert!(!stem_is_ambiguous(&e, "sample", &only));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&e);
    }
}
