//! Drive a real search through the supervisor, with no window involved.
//!
//! This is the milestone-1 acceptance criterion expressed as a test: spawn the
//! engine, watch the artifact reports appear, and read the results back. It covers
//! the parts a person clicking buttons would exercise, minus the buttons.
//!
//! Skipped unless both are set, so it never fails a machine that has no engine:
//!
//!   MUMDIA_BIN        the engine binary
//!   MUMDIA_TEST_MZML  an mzML to search
//!   MUMDIA_TEST_FASTA a FASTA to digest
//!
//! `ci/smoke.sh` generates a suitable fixture pair in its work directory.

use std::time::{Duration, Instant};

/// Poll a run until it leaves the running state, or give up.
fn wait_for_finish(
    run: &mumdia_console::run::Run,
    limit: Duration,
) -> mumdia_console::run::Snapshot {
    let start = Instant::now();
    loop {
        let s = run.snapshot();
        if s.status != "running" && s.status != "starting" {
            return s;
        }
        if start.elapsed() > limit {
            panic!(
                "run did not finish within {limit:?}; last status {}",
                s.status
            );
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

#[test]
fn a_fasta_search_runs_to_completion_and_reports_itself() {
    let (Some(mzml), Some(fasta)) = (env("MUMDIA_TEST_MZML"), env("MUMDIA_TEST_FASTA")) else {
        eprintln!("MUMDIA_TEST_MZML / MUMDIA_TEST_FASTA not set; skipping");
        return;
    };
    if env("MUMDIA_BIN").is_none() {
        eprintln!("MUMDIA_BIN not set; skipping");
        return;
    }

    let out = std::env::temp_dir().join(format!("mumdia_console_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);

    let req = mumdia_console::run::Request {
        mzml,
        out_dir: out.display().to_string(),
        fasta: Some(fasta),
        lib_precursors: None,
        lib_fragments: None,
        config: None,
        threads: Some(2),
    };

    let run = mumdia_console::run::start("e2e".into(), req).expect("the engine should start");

    // The displayed command must be the real invocation, because a user is invited
    // to copy it into a terminal.
    let cmd = run.snapshot().command;
    assert!(cmd.contains(" run "), "{cmd}");
    assert!(cmd.contains("--fasta"), "{cmd}");
    assert!(cmd.contains("--threads 2"), "{cmd}");

    let s = wait_for_finish(&run, Duration::from_secs(600));
    assert_eq!(
        s.status,
        "done",
        "run failed: {:?}\n{}",
        s.error,
        s.log.join("\n")
    );
    assert_eq!(s.exit_code, Some(0));

    // Progress was actually observed while it ran, not reconstructed at the end.
    let names: Vec<&str> = s.stages.iter().map(|x| x.name.as_str()).collect();
    for expected in ["convert", "search-seed", "extract", "rescore", "quant"] {
        assert!(
            names.contains(&expected),
            "missing stage {expected}: {names:?}"
        );
    }

    // The log is captured from the engine's stderr.
    assert!(!s.log.is_empty(), "no log lines were captured");

    // Results come from the scored table's own report.
    let r = s
        .results
        .expect("results should be present after a successful run");
    assert!(!r.classifier.is_empty(), "the classifier should be named");
    assert!(r.psms > 0, "a successful fixture run scores some PSMs");
    assert!(
        r.has_peptides_tsv,
        "the report stage should have written peptides.tsv"
    );

    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn stopping_a_run_kills_it_and_leaves_no_temp_files() {
    let (Some(mzml), Some(fasta)) = (env("MUMDIA_TEST_MZML"), env("MUMDIA_TEST_FASTA")) else {
        eprintln!("MUMDIA_TEST_MZML / MUMDIA_TEST_FASTA not set; skipping");
        return;
    };
    if env("MUMDIA_BIN").is_none() {
        eprintln!("MUMDIA_BIN not set; skipping");
        return;
    }

    let out = std::env::temp_dir().join(format!("mumdia_console_cancel_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);

    let req = mumdia_console::run::Request {
        mzml,
        out_dir: out.display().to_string(),
        fasta: Some(fasta),
        lib_precursors: None,
        lib_fragments: None,
        config: None,
        threads: Some(1),
    };
    let run = mumdia_console::run::start("cancel".into(), req).expect("the engine should start");

    // Long enough to be doing real work and holding files open, short enough that a
    // small fixture has not finished.
    std::thread::sleep(Duration::from_millis(400));
    run.cancel();

    let s = wait_for_finish(&run, Duration::from_secs(60));

    // The fixture search takes about two seconds, so on a fast machine it can finish
    // before the cancel lands. Say so rather than asserting something that did not
    // happen: a test that quietly passes without exercising the thing it names is
    // worse than no test. Point it at real data and the cancellation path runs.
    if s.status == "done" {
        eprintln!(
            "the run completed in {} ms, before the stop could land;              cancellation was NOT exercised (use a larger input to cover it)",
            s.elapsed_ms
        );
    } else {
        assert_eq!(
            s.status, "cancelled",
            "a stopped run reports itself as cancelled"
        );
        assert_ne!(s.exit_code, Some(0), "a killed engine did not exit cleanly");
    }

    // A hard kill skips destructors, so the sweep is what keeps the folder clean.
    let mut leftovers = Vec::new();
    let mut stack = vec![out.clone()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.to_string_lossy().contains(".tmp-") {
                leftovers.push(p);
            }
        }
    }
    assert!(
        leftovers.is_empty(),
        "temp files survived the sweep: {leftovers:?}"
    );

    let _ = std::fs::remove_dir_all(&out);
}

/// Create the primary environment for real and check every role can import.
///
/// This is the milestone-2 acceptance criterion: no conda, no pre-existing Python,
/// just the bundled installer and the pinned requirements. Opt in with
/// MUMDIA_TEST_INSTALL=1, because it downloads several hundred megabytes.
#[test]
fn the_primary_environment_installs_and_imports() {
    if std::env::var("MUMDIA_TEST_INSTALL").ok().as_deref() != Some("1") {
        eprintln!("MUMDIA_TEST_INSTALL=1 not set; skipping the real installation");
        return;
    }
    use mumdia_console::components::{self, Env};

    assert!(
        components::find_uv().is_some(),
        "uv must be bundled or on PATH for the installer to work"
    );
    assert!(
        components::requirements(Env::Primary).is_ok(),
        "the compiled-in requirements must be writable to the data directory"
    );

    let installer = std::sync::Arc::new(components::Installer::default());
    components::install(std::sync::Arc::clone(&installer), Env::Primary)
        .expect("the installation should start");

    let start = Instant::now();
    loop {
        let s = installer.refresh(Env::Primary);
        if s.install_status == "done" {
            assert!(s.complete, "installed but not importable: {:?}", s.missing);
            assert!(
                s.versions.contains_key("torch") && s.versions.contains_key("deeplc"),
                "the versions that change results should be reported: {:?}",
                s.versions
            );
            break;
        }
        if s.install_status == "failed" {
            panic!(
                "installation failed: {}\n{}",
                s.error.unwrap_or_default(),
                s.install_log.join("\n")
            );
        }
        assert!(
            start.elapsed() < Duration::from_secs(1800),
            "installation did not finish within 30 minutes"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
}
