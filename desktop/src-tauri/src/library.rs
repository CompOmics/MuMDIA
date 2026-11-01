//! A spectral library built without a search (issue #233): what the engine's
//! `build-library` reported, and copying a library to a folder the user chooses.
//!
//! The engine writes `library.json` beside the two tables. It is the one record of how
//! the library was built and how to search with it, so the interface reads it rather than
//! keeping its own: the counts and models on the result card, and the
//! `rt_im_train.library_irt` value a library-mode search of these tables needs to decide
//! its retention-time handling as the FASTA search would have.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// The file the engine describes a built library in.
pub const LIBRARY_JSON: &str = "library.json";

/// The `library.json` a library build wrote into `dir`, if it is there and readable.
pub fn read(dir: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(dir.join(LIBRARY_JSON)).ok()?).ok()
}

/// The `library.json` beside `precursors`, when it describes that precursor table.
///
/// Matched on the file name, not the whole path, so a library copied to another folder
/// (`copy_to`, which rewrites the paths, or by hand, which does not) is still described.
pub fn info_for(precursors: &Path) -> Option<Value> {
    let v = read(precursors.parent()?)?;
    let named = v.get("precursors")?.get("path")?.as_str()?;
    (Path::new(named).file_name()? == precursors.file_name()?).then_some(v)
}

/// The files of a library that exist: the two tables, the report beside each, and the
/// `library.json` that describes them.
pub fn files_of(precursors: &Path, fragments: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for t in [precursors, fragments] {
        out.push(t.to_path_buf());
        let mut report = t.as_os_str().to_owned();
        report.push(".report.json");
        out.push(PathBuf::from(report));
    }
    if info_for(precursors).is_some() {
        if let Some(dir) = precursors.parent() {
            out.push(dir.join(LIBRARY_JSON));
        }
    }
    out.into_iter().filter(|p| p.is_file()).collect()
}

/// Copy a library into `dest` and return the paths of the two tables there.
///
/// Refuses to overwrite: a folder that already holds a file of the same name may hold
/// another library, and replacing one table of a pair silently would mismatch them. The
/// copied `library.json` names the copies, so the folder describes itself.
pub fn copy_to(
    precursors: &Path,
    fragments: &Path,
    dest: &Path,
) -> Result<(String, String), String> {
    for t in [precursors, fragments] {
        if !t.is_file() {
            return Err(format!("{} does not exist", t.display()));
        }
    }
    std::fs::create_dir_all(dest)
        .map_err(|e| format!("cannot create the folder {}: {e}", dest.display()))?;
    let same_dir = |a: &Path| {
        a.parent()
            .and_then(|p| std::fs::canonicalize(p).ok())
            .zip(std::fs::canonicalize(dest).ok())
            .is_some_and(|(a, b)| a == b)
    };
    if same_dir(precursors) || same_dir(fragments) {
        return Err(format!("the library is already in {}", dest.display()));
    }
    let files = files_of(precursors, fragments);
    let target = |f: &Path| dest.join(f.file_name().unwrap_or_default());
    let taken: Vec<String> = files
        .iter()
        .map(|f| target(f))
        .filter(|t| t.exists())
        .map(|t| t.display().to_string())
        .collect();
    if !taken.is_empty() {
        return Err(format!(
            "{} already holds {}; choose an empty folder, so the copy cannot be mixed with \
             another library",
            dest.display(),
            taken.join(", ")
        ));
    }
    let (new_p, new_f) = (target(precursors), target(fragments));
    for f in &files {
        let to = target(f);
        if f.file_name().is_some_and(|n| n == LIBRARY_JSON) {
            // Every mention of the two tables, as the engine spelled them (the reuse
            // commands included), now names the copies.
            let text = std::fs::read_to_string(f)
                .map_err(|e| format!("cannot read {}: {e}", f.display()))?;
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| format!("{} is not valid JSON: {e}", f.display()))?;
            let named = |k: &str| {
                v.get(k)
                    .and_then(|t| t.get("path"))
                    .and_then(|p| p.as_str())
                    .map(|p| json_escaped(Path::new(p)))
            };
            let mut moved = text.clone();
            for (old, new) in [(named("precursors"), &new_p), (named("fragments"), &new_f)] {
                if let Some(old) = old.filter(|o| !o.is_empty()) {
                    moved = moved.replace(&old, &json_escaped(new));
                }
            }
            std::fs::write(&to, moved)
        } else {
            std::fs::copy(f, &to).map(|_| ())
        }
        .map_err(|e| format!("cannot copy {} to {}: {e}", f.display(), to.display()))?;
    }
    Ok((new_p.display().to_string(), new_f.display().to_string()))
}

/// A path as it appears inside a JSON string, so it can be found in the file's text.
fn json_escaped(p: &Path) -> String {
    let s = serde_json::to_string(&p.display().to_string()).unwrap_or_default();
    s.trim_matches('"').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("mumdia_console_lib_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A built library as `build-library` leaves it: two tables, their reports and
    /// `library.json` naming them.
    fn built(dir: &Path) -> (PathBuf, PathBuf) {
        let p = dir.join("fragment_library_precursors.parquet");
        let f = dir.join("fragment_library_fragments.parquet");
        for t in [&p, &f] {
            std::fs::write(t, b"PAR1").unwrap();
            std::fs::write(format!("{}.report.json", t.display()), "{}").unwrap();
        }
        let info = serde_json::json!({
            "precursors": {"path": p.display().to_string(), "rows": 3},
            "fragments": {"path": f.display().to_string(), "rows": 9},
            "reuse": {
                "lib_precursors": p.display().to_string(),
                "lib_fragments": f.display().to_string(),
                "library_irt": "deeplc",
            },
        });
        std::fs::write(dir.join(LIBRARY_JSON), info.to_string()).unwrap();
        (p, f)
    }

    #[test]
    fn library_json_describes_only_the_table_it_names() {
        let dir = scratch("info");
        let (p, f) = built(&dir);
        assert_eq!(info_for(&p).unwrap()["reuse"]["library_irt"], "deeplc");
        // Another table in the same folder is not the library it describes.
        assert!(info_for(&f).is_none());
        assert!(info_for(&dir.join("other.parquet")).is_none());
        assert_eq!(files_of(&p, &f).len(), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_copy_names_its_own_tables_and_never_overwrites() {
        let root = scratch("copy");
        let src = root.join("src");
        std::fs::create_dir_all(&src).unwrap();
        let (p, f) = built(&src);
        let dest = root.join("dest");
        let (np, nf) = copy_to(&p, &f, &dest).unwrap();
        assert!(Path::new(&np).is_file() && Path::new(&nf).is_file());
        assert!(dest
            .join("fragment_library_fragments.parquet.report.json")
            .is_file());
        let info = info_for(Path::new(&np)).expect("the copy is described");
        assert_eq!(info["precursors"]["path"], np.as_str());
        assert_eq!(info["reuse"]["lib_fragments"], nf.as_str());
        // A second copy into the same folder is refused, and so is a copy onto itself.
        let e = copy_to(&p, &f, &dest).unwrap_err();
        assert!(e.contains("already holds"), "{e}");
        let e = copy_to(&p, &f, &src).unwrap_err();
        assert!(e.contains("already in"), "{e}");
        // A library without library.json (a DIA-NN build) copies its tables alone.
        let bare = root.join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        let (bp, bf) = (
            bare.join("lib_precursors.parquet"),
            bare.join("lib_fragments.parquet"),
        );
        std::fs::write(&bp, b"PAR1").unwrap();
        std::fs::write(&bf, b"PAR1").unwrap();
        let out = root.join("bare_copy");
        copy_to(&bp, &bf, &out).unwrap();
        assert_eq!(std::fs::read_dir(&out).unwrap().count(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }
}
