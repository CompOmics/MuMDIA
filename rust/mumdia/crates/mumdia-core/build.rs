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

    // Re-run when the checked-out commit changes, so the stamp does not go stale in an
    // incremental build.
    //
    // This used to guess relative paths (`../../../.git/HEAD`, `../../.git/HEAD`). A
    // build script runs with its CWD at the CRATE root, and this crate is
    // `rust/mumdia/crates/mumdia-core`, so the repository's `.git` is FOUR levels up:
    // both guesses resolved to `rust/.git` and `rust/mumdia/.git`, neither of which
    // exists. Every path failed the `exists()` test, no `rerun-if-changed` was emitted
    // for git state at all, and the stamp then survived arbitrarily many commits.
    //
    // That was not cosmetic. A benchmark run on this branch recorded
    // `git_sha = c13a623a8d33-dirty` in its manifest while HEAD was nine commits later
    // at `9fc1c06`, and another build stamped a sha that is not a valid object in this
    // repository at all. Provenance that silently names the wrong commit is worse than
    // none, because the manifest is what a result is supposed to be traceable through.
    //
    // Ask git for the directory instead of guessing at it. `--absolute-git-dir` also
    // handles a worktree or a submodule, where `.git` is a file rather than a directory.
    if let Some(gitdir) = git(&["rev-parse", "--absolute-git-dir"]) {
        let gitdir = std::path::Path::new(&gitdir);
        println!("cargo:rerun-if-changed={}", gitdir.join("HEAD").display());
        // The commit a branch points at changes the ref file, not HEAD, so watch both.
        if let Some(head_ref) = git(&["symbolic-ref", "--quiet", "HEAD"]) {
            let refpath = gitdir.join(&head_ref);
            if refpath.exists() {
                println!("cargo:rerun-if-changed={}", refpath.display());
            }
            // A packed ref has no loose file; `packed-refs` is where it moves.
            let packed = gitdir.join("packed-refs");
            if packed.exists() {
                println!("cargo:rerun-if-changed={}", packed.display());
            }
        }
        // The index is what `git status --porcelain` compares against, so it is the
        // closest available proxy for the `-dirty` marker.
        let index = gitdir.join("index");
        if index.exists() {
            println!("cargo:rerun-if-changed={}", index.display());
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
