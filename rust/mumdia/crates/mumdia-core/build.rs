//! Stamp the build's source identity into the binary.
//!
//! `manifest.json` recorded only `mumdia_version`, which is `0.1.0` for every
//! build ever made from this branch. A benchmark record is supposed to carry the
//! commit it came from (docs/20_sensitivity_and_quantification_playbook.md asks
//! for "commit/build:"), and the manifest could not supply one, so a result could
//! not be tied back to the code that produced it.
//!
//! The commit DATE is used rather than the build time on purpose: the build time
//! differs on every rebuild, which would make the binary unreproducible and defeat
//! caching for no benefit. The commit date answers the question a reader actually
//! has, which is how old the code is.
//!
//! When git is unavailable, as in a build from a release tarball, the values become
//! `unknown` rather than failing the build.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn main() {
    let sha = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());

    // A dirty worktree must be visible: a number produced from uncommitted code is
    // not reproducible from the named commit, and saying so is the whole point.
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(s) if !s.is_empty() => true,
        Some(_) => false,
        // Unknown git state is not the same as clean; do not claim clean.
        None => false,
    };
    let sha = if dirty && sha != "unknown" {
        format!("{sha}-dirty")
    } else {
        sha
    };

    let date = git(&["log", "-1", "--format=%cI"]).unwrap_or_else(|| "unknown".into());

    println!("cargo:rustc-env=MUMDIA_GIT_SHA={sha}");
    println!("cargo:rustc-env=MUMDIA_COMMIT_DATE={date}");

    // Re-run when the checked-out commit or the index changes, so the stamp does
    // not go stale in an incremental build. Paths are relative to this crate.
    for p in [
        "../../../.git/HEAD",
        "../../../.git/index",
        "../../.git/HEAD",
        "../../.git/index",
    ] {
        if std::path::Path::new(p).exists() {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
