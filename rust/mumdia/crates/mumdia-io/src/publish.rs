//! Atomic publication of output files: write to a sibling temp name, then rename onto
//! the final path. Every Parquet writer in [`crate::table`], `json::write_json` and the
//! TSV reports publish through [`AtomicPath`].

use anyhow::{Context, Result};

/// A sibling temp path for an artifact, and the rename that publishes it.
///
/// Every artifact used to be written with `File::create` directly AT its final path,
/// which truncates on open. Three consequences, all real:
///
/// - a rerun destroyed the previous good artifact BEFORE producing a replacement, so
///   an interrupted rerun left neither;
/// - Ctrl-C or a crash part-way through a multi-gigabyte write left rubble under the
///   canonical name. The parquet footer is written last and every reader requires it,
///   so such a file fails to open rather than reading as a short table, but a
///   directory of unopenable artifacts is still worse than a directory of intact ones;
/// - nothing distinguished "this run wrote it" from "a previous run left it".
///
/// Writing to `<path>.tmp-<pid>-<n>` and renaming on success addresses all three: the
/// rename is atomic on both POSIX and Windows for a same-directory target, so a reader
/// sees either the old artifact or the new one, never a partial one. A killed run
/// leaves at most a recognisable `.tmp-<pid>-<n>` file, which is inert.
///
/// Two guarantees this makes, and one it does not (docs/29 #4): the destination is never
/// removed before the rename, so a failed publication leaves the previous artifact in
/// place; and `n` is a process-wide counter, so two writers for one destination in one
/// process cannot share a temporary file. It does not promise that the rename succeeds
/// while another Windows process holds the destination open without delete sharing;
/// then the rename fails and the old file stays, which is the first guarantee at work.
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct AtomicPath {
    tmp: std::path::PathBuf,
    final_path: std::path::PathBuf,
    published: bool,
}

impl AtomicPath {
    pub fn new(path: &str) -> Result<AtomicPath> {
        let final_path = std::path::PathBuf::from(path);
        if let Some(parent) = final_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating output directory {}", parent.display()))?;
            }
        }
        let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = std::path::PathBuf::from(format!("{path}.tmp-{}-{n}", std::process::id()));
        Ok(AtomicPath {
            tmp,
            final_path,
            published: false,
        })
    }

    pub fn tmp(&self) -> &std::path::Path {
        &self.tmp
    }

    /// Move the completed temp file onto the final path.
    pub fn publish(mut self) -> Result<()> {
        // `std::fs::rename` replaces an existing destination FILE on POSIX and, through
        // `MoveFileExW(MOVEFILE_REPLACE_EXISTING)`, on Windows, so the destination is not
        // removed first. Removing it first meant a rename that then failed had already
        // destroyed the previous result, and gave every reader a window with no file at
        // the final path at all (docs/29 #4). Now a failed rename is an error with the
        // previous artifact still where it was.
        std::fs::rename(&self.tmp, &self.final_path).with_context(|| {
            format!(
                "publishing {} -> {}",
                self.tmp.display(),
                self.final_path.display()
            )
        })?;
        self.published = true;
        Ok(())
    }
}

impl Drop for AtomicPath {
    fn drop(&mut self) {
        // Abandoned write (an error return or an unwind): take the rubble with us so
        // the output directory does not accumulate temp files.
        if !self.published {
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// How [`publish_copy_of`] put the bytes at the destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileCopy {
    /// A second directory entry for the same file: no byte was read or written.
    HardLink,
    /// The filesystem refused the link, so the bytes were copied.
    ByteCopy,
}

impl FileCopy {
    /// The spelling a report records.
    pub fn as_str(self) -> &'static str {
        match self {
            FileCopy::HardLink => "hard_link",
            FileCopy::ByteCopy => "byte_copy",
        }
    }
}

/// Publish the bytes of `src` at `out`, unchanged and without decoding them.
///
/// A hard link into the [`AtomicPath`] temp name, then the usual rename, so a reader of
/// `out` sees its previous content or the complete new one and never a partial file. A
/// filesystem that cannot link (another volume, FAT, some network and sync folders) falls
/// back to a byte copy into the same temp name. An error here leaves `out` as it was.
///
/// The two names share one file after a link. Every writer in this crate (the parquet
/// writers, [`write_batches`](crate::table::write_batches), this function and
/// `json::write_json`) publishes by renaming a new file over its destination, which
/// replaces the directory entry and leaves the other name's file alone, so rewriting either
/// artifact through them never changes the other.
/// That does not hold for a writer that opens its destination with `File::create`, which
/// truncates the shared file and so writes through both names. The engine has two, the
/// `features` PIN and rescore's tab-separated handoff, and neither writes a parquet
/// artifact path. A tool that edits one of the linked files IN PLACE changes both.
pub fn publish_copy_of(src: &str, out: &str) -> Result<FileCopy> {
    publish_copy_of_with(src, out, true)
}

/// [`publish_copy_of`] without the hard link: `out` is always a file of its own, so a tool
/// that edits either name in place cannot change the other. For a copy that other runs
/// read later, such as a stored library.
pub fn publish_byte_copy(src: &str, out: &str) -> Result<()> {
    publish_copy_of_with(src, out, false).map(|_| ())
}

fn publish_copy_of_with(src: &str, out: &str, allow_link: bool) -> Result<FileCopy> {
    let target = AtomicPath::new(out)?;
    let tmp = target.tmp().to_path_buf();
    // A temp name left by a killed process with the same pid and counter would make the
    // link fail, and a byte copy onto it would write THROUGH it if it is itself a link to
    // `src`, truncating the source. Start both from a fresh name.
    let _ = std::fs::remove_file(&tmp);
    let linked = allow_link && std::fs::hard_link(src, &tmp).is_ok();
    let how = if linked {
        FileCopy::HardLink
    } else {
        std::fs::copy(src, &tmp).with_context(|| format!("copying {src} -> {}", tmp.display()))?;
        FileCopy::ByteCopy
    };
    target.publish()?;
    // POSIX `rename` does nothing and succeeds when both names already refer to the same
    // file, which is the case when `out` is a link to `src` from a previous run. The temp
    // name then survives the rename; it is a third name for that same file, so removing it
    // loses nothing.
    if tmp.exists() {
        let _ = std::fs::remove_file(&tmp);
    }
    Ok(how)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mumdia_atomic_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn two_writers_for_one_destination_get_different_temporary_files() {
        let d = dir("two_writers");
        let final_path = d.join("out.parquet").to_str().unwrap().to_string();
        let a = AtomicPath::new(&final_path).unwrap();
        let b = AtomicPath::new(&final_path).unwrap();
        assert_ne!(
            a.tmp(),
            b.tmp(),
            "the suffix must not be the process id alone"
        );
        drop(a);
        drop(b);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failed_publication_leaves_the_previous_result_in_place() {
        // The destination is a non-empty directory, which a file cannot be renamed onto
        // on any platform, so publication fails. Before the fix the destination was
        // removed first and the failure left nothing behind.
        let d = dir("failed_publish");
        let final_path = d.join("out.parquet");
        std::fs::create_dir_all(&final_path).unwrap();
        std::fs::write(final_path.join("previous"), b"previous result").unwrap();
        let ap = AtomicPath::new(final_path.to_str().unwrap()).unwrap();
        std::fs::write(ap.tmp(), b"new result").unwrap();
        assert!(
            ap.publish().is_err(),
            "renaming a file onto a directory must fail"
        );
        assert!(
            final_path.join("previous").is_file(),
            "the previous result must survive a failed publication"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_successful_publication_replaces_the_previous_file() {
        let d = dir("replace");
        let final_path = d.join("out.parquet");
        let fp = final_path.to_str().unwrap().to_string();
        let first = AtomicPath::new(&fp).unwrap();
        std::fs::write(first.tmp(), b"v1").unwrap();
        first.publish().unwrap();
        let second = AtomicPath::new(&fp).unwrap();
        std::fs::write(second.tmp(), b"v2").unwrap();
        second.publish().unwrap();
        assert_eq!(std::fs::read(&final_path).unwrap(), b"v2");
        assert!(
            std::fs::read_dir(&d).unwrap().count() == 1,
            "no temporary file may remain after publication"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_published_copy_is_the_source_bytes_by_link_and_by_copy() {
        let d = dir("copy_of");
        let src = d.join("features.parquet");
        std::fs::write(&src, b"PAR1 the source bytes PAR1").unwrap();
        let src_s = src.to_str().unwrap();
        for (allow_link, want) in [(true, FileCopy::HardLink), (false, FileCopy::ByteCopy)] {
            let out = d.join(format!("competed_{allow_link}.parquet"));
            let out_s = out.to_str().unwrap();
            // Twice: the second publication lands on a destination that already holds the
            // same bytes, and for the link case already IS the same file.
            for _ in 0..2 {
                let how = publish_copy_of_with(src_s, out_s, allow_link).unwrap();
                // A filesystem without hard links degrades to the copy; both are correct.
                assert!(how == want || how == FileCopy::ByteCopy, "{how:?}");
                assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(&src).unwrap());
            }
        }
        // The source is intact, and no temporary name survived either publication: the
        // second link publication renamed a link onto a link to the same file, which POSIX
        // turns into a no-op that leaves the temporary name behind.
        assert_eq!(std::fs::read(&src).unwrap(), b"PAR1 the source bytes PAR1");
        let names: Vec<String> = std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().all(|n| !n.contains(".tmp-")),
            "a temporary file survived: {names:?}"
        );
        assert_eq!(names.len(), 3, "{names:?}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn republishing_either_name_leaves_the_other_file_alone() {
        // After a link the two names share one file. The writers publish by renaming a new
        // file over the destination, so rewriting the source must not change the copy.
        let d = dir("link_independence");
        let src = d.join("features.parquet");
        let out = d.join("competed.parquet");
        std::fs::write(&src, b"v1").unwrap();
        publish_copy_of(src.to_str().unwrap(), out.to_str().unwrap()).unwrap();
        let rewrite = AtomicPath::new(src.to_str().unwrap()).unwrap();
        std::fs::write(rewrite.tmp(), b"v2").unwrap();
        rewrite.publish().unwrap();
        assert_eq!(std::fs::read(&src).unwrap(), b"v2");
        assert_eq!(std::fs::read(&out).unwrap(), b"v1");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failed_copy_leaves_the_destination_as_it_was() {
        let d = dir("copy_fails");
        let out = d.join("competed.parquet");
        std::fs::write(&out, b"previous").unwrap();
        let missing = d.join("no_such_features.parquet");
        assert!(publish_copy_of(missing.to_str().unwrap(), out.to_str().unwrap()).is_err());
        assert_eq!(std::fs::read(&out).unwrap(), b"previous");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }
}
