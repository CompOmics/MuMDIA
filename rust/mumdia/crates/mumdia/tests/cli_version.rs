//! The built binary, run as a subprocess.
//!
//! The debug build overflowed its 1 MiB Windows main-thread stack on `--version`, before
//! printing anything (docs/30 R9). No library test could see that: it is a property of the
//! binary's entry point under the developer profile, so it is checked here on whichever
//! profile `cargo test` builds.

use std::process::Command;

fn mumdia() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mumdia"))
}

#[test]
fn the_built_binary_prints_its_version() {
    let out = mumdia()
        .arg("--version")
        .output()
        .expect("run mumdia --version");
    assert!(
        out.status.success(),
        "status {:?}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains(env!("CARGO_PKG_VERSION")),
        "expected the crate version in {text:?}"
    );
}

#[test]
fn the_built_binary_prints_its_help() {
    let out = mumdia().arg("--help").output().expect("run mumdia --help");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("run-experiment"), "{text}");
}
