//! The sample sheet of `run-experiment --sample-sheet` (CompOmics/MuMDIA#275).
//!
//! A tab-separated table with a header row. `file` (required) names a run's input, a path
//! relative to the sheet's directory or absolute; `name` (optional) is the run's name, which
//! passes the rules of `--run-names`; `condition` (optional) is the run's condition, which
//! `quant.normalisation_scope = condition` normalises within. Header names are compared
//! without regard to case; blank lines and lines that start with `#` are skipped, a UTF-8
//! byte-order mark and Windows line ends are accepted, and a column the sheet does not know
//! is ignored with a warning.
//!
//! Without `--mzml` the sheet is the input list, in its row order. With `--mzml` each input
//! is matched to the row whose `file` is the same path or, failing that, has the same file
//! name (without regard to case); the sheet then supplies names and conditions only. An
//! explicit `--run-names` or `--condition` wins over the sheet's column.

use std::path::Path;

use anyhow::{Context, Result};
use tracing::warn;

/// One row of a sample sheet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SheetRow {
    /// The input as the sheet names it.
    pub given: String,
    /// The input resolved against the sheet's directory.
    pub file: String,
    pub name: Option<String>,
    pub condition: Option<String>,
}

/// Parse a sample sheet's text; `dir` is the directory relative `file` entries resolve
/// against, `origin` names the sheet in errors.
pub fn parse(text: &str, dir: &Path, origin: &str) -> Result<Vec<SheetRow>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text
        .split('\n')
        .enumerate()
        .map(|(i, l)| (i + 1, l.strip_suffix('\r').unwrap_or(l)))
        .filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
    let Some((_, header)) = lines.next() else {
        anyhow::bail!(
            "{origin}: the sample sheet is empty; it needs a header row with a `file` column"
        );
    };
    let cols: Vec<String> = header
        .split('\t')
        .map(|c| c.trim().to_ascii_lowercase())
        .collect();
    let find = |want: &str| -> Result<Option<usize>> {
        let at: Vec<usize> = (0..cols.len()).filter(|&i| cols[i] == want).collect();
        match at.as_slice() {
            [] => Ok(None),
            [i] => Ok(Some(*i)),
            _ => anyhow::bail!("{origin}: the column `{want}` appears {} times", at.len()),
        }
    };
    let Some(file_col) = find("file")? else {
        anyhow::bail!(
            "{origin}: the sample sheet has no `file` column (columns: {}); it needs `file`, \
             and may have `name` and `condition`, separated by tabs",
            cols.join(", ")
        );
    };
    let name_col = find("name")?;
    let cond_col = find("condition")?;
    let unknown: Vec<&String> = cols
        .iter()
        .filter(|c| !matches!(c.as_str(), "file" | "name" | "condition"))
        .collect();
    if !unknown.is_empty() {
        warn!(columns = ?unknown, "{origin}: sample-sheet columns other than file, name and condition are ignored");
    }
    let mut rows: Vec<SheetRow> = Vec::new();
    for (line_no, line) in lines {
        let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
        if fields.len() > cols.len() {
            anyhow::bail!(
                "{origin}, line {line_no}: {} fields under a header of {} columns",
                fields.len(),
                cols.len()
            );
        }
        let field = |i: Option<usize>| {
            i.and_then(|i| fields.get(i))
                .copied()
                .filter(|v| !v.is_empty())
                .map(str::to_string)
        };
        let Some(given) = field(Some(file_col)) else {
            anyhow::bail!("{origin}, line {line_no}: the `file` field is empty");
        };
        let file = if Path::new(&given).is_absolute() {
            given.clone()
        } else {
            dir.join(&given).to_string_lossy().into_owned()
        };
        if let Some(prev) = rows.iter().find(|r| same_path(&r.file, &file)) {
            anyhow::bail!(
                "{origin}, line {line_no}: {given:?} is listed twice (as {:?} before); each \
                 input is one run",
                prev.given
            );
        }
        rows.push(SheetRow {
            given,
            file,
            name: field(name_col),
            condition: field(cond_col),
        });
    }
    if rows.is_empty() {
        anyhow::bail!("{origin}: the sample sheet has a header and no run");
    }
    Ok(rows)
}

/// Read and parse the sample sheet at `path`.
pub fn read(path: &str) -> Result<Vec<SheetRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the sample sheet {path}"))?;
    let dir = Path::new(path)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    parse(&text, &dir, path)
}

/// The inputs, run names and conditions of an experiment after the sample sheet.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resolved {
    pub mzml: Vec<String>,
    /// `None`: derive the names from the inputs.
    pub run_names: Option<Vec<String>>,
    /// One per input (empty for a run without one), or `None` when no input has one.
    pub conditions: Option<Vec<String>>,
}

/// Combine `--mzml`, `--run-names` and `--condition` with the sample sheet: an explicit flag
/// wins over the sheet's column, and without `--mzml` the sheet lists the inputs.
pub fn resolve(
    mzml: &[String],
    run_names: &[String],
    conditions: &[String],
    sheet: Option<&[SheetRow]>,
) -> Result<Resolved> {
    let some = |v: &[String]| (!v.is_empty()).then(|| v.to_vec());
    let Some(rows) = sheet else {
        return Ok(Resolved {
            mzml: mzml.to_vec(),
            run_names: some(run_names),
            conditions: some(conditions),
        });
    };
    // The sheet row of each input.
    let matched: Vec<Option<&SheetRow>> = if mzml.is_empty() {
        rows.iter().map(Some).collect()
    } else {
        let mut used = vec![false; rows.len()];
        let mut out = Vec::with_capacity(mzml.len());
        for m in mzml {
            let by_path: Vec<usize> = (0..rows.len())
                .filter(|&i| same_path(&rows[i].file, m) || same_path(&rows[i].given, m))
                .collect();
            let hits = if by_path.is_empty() {
                let want = file_name(m).to_lowercase();
                (0..rows.len())
                    .filter(|&i| file_name(&rows[i].given).to_lowercase() == want)
                    .collect()
            } else {
                by_path
            };
            match hits.as_slice() {
                [] => out.push(None),
                [i] => {
                    used[*i] = true;
                    out.push(Some(&rows[*i]));
                }
                _ => anyhow::bail!(
                    "--sample-sheet: --mzml {m:?} matches {} rows of the sheet by file name; \
                     give the rows' paths so that each input matches one row",
                    hits.len()
                ),
            }
        }
        let unmatched: Vec<&str> = rows
            .iter()
            .zip(&used)
            .filter(|(_, u)| !**u)
            .map(|(r, _)| r.given.as_str())
            .collect();
        if !unmatched.is_empty() {
            warn!(files = ?unmatched, "--sample-sheet lists files that are not among the --mzml inputs; they are ignored");
        }
        out
    };
    let mzml_out: Vec<String> = if mzml.is_empty() {
        rows.iter().map(|r| r.file.clone()).collect()
    } else {
        mzml.to_vec()
    };
    let inputs = |i: usize| mzml_out[i].as_str();
    let names = if !run_names.is_empty() {
        Some(run_names.to_vec())
    } else if matched.iter().any(|r| r.is_some_and(|r| r.name.is_some())) {
        let missing: Vec<&str> = (0..matched.len())
            .filter(|&i| matched[i].and_then(|r| r.name.as_ref()).is_none())
            .map(inputs)
            .collect();
        if !missing.is_empty() {
            anyhow::bail!(
                "--sample-sheet names some runs and not others; give every run a name or \
                 none. Without a name: {missing:?}"
            );
        }
        Some(
            matched
                .iter()
                .map(|r| r.and_then(|r| r.name.clone()).unwrap_or_default())
                .collect(),
        )
    } else {
        None
    };
    let conds = if !conditions.is_empty() {
        Some(conditions.to_vec())
    } else if matched
        .iter()
        .any(|r| r.is_some_and(|r| r.condition.is_some()))
    {
        Some(
            matched
                .iter()
                .map(|r| r.and_then(|r| r.condition.clone()).unwrap_or_default())
                .collect(),
        )
    } else {
        None
    };
    Ok(Resolved {
        mzml: mzml_out,
        run_names: names,
        conditions: conds,
    })
}

fn file_name(p: &str) -> &str {
    p.rsplit(['/', '\\']).next().unwrap_or(p)
}

fn same_path(a: &str, b: &str) -> bool {
    let norm = |s: &str| s.replace('\\', "/");
    norm(a) == norm(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn a_sheet_with_a_bom_crlf_comments_and_relative_paths_parses() {
        let text = "\u{feff}File\tName\tCondition\tnotes\r\n# a comment\r\n\r\nA01.mzML\tA01\tA\tx\r\n/data/B01.raw\t\tB\r\n";
        let rows = parse(text, Path::new("/sheets"), "s.tsv").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].given, "A01.mzML");
        assert_eq!(
            Path::new(&rows[0].file),
            Path::new("/sheets").join("A01.mzML")
        );
        assert_eq!(rows[0].name.as_deref(), Some("A01"));
        assert_eq!(rows[0].condition.as_deref(), Some("A"));
        assert_eq!(rows[1].name, None);
        assert_eq!(rows[1].condition.as_deref(), Some("B"));
        if cfg!(unix) {
            assert_eq!(rows[1].file, "/data/B01.raw");
        }
    }

    #[test]
    fn bad_sheets_are_refused_with_the_reason() {
        let dir = Path::new("d");
        for (text, why) in [
            ("", "empty"),
            ("name\tcondition\na\tA\n", "no `file` column"),
            ("file\tfile\na\tb\n", "appears 2 times"),
            ("file\n", "no run"),
            ("file\tcondition\na.mzML\tA\tB\n", "3 fields"),
            ("file\tcondition\n\tA\n", "`file` field is empty"),
            ("file\na.mzML\nb.mzML\na.mzML\n", "listed twice"),
        ] {
            let e = parse(text, dir, "s.tsv").unwrap_err().to_string();
            assert!(e.contains(why), "{text:?}: {e}");
        }
    }

    #[test]
    fn the_sheet_lists_the_inputs_or_labels_the_given_ones() {
        let rows = parse(
            "file\tname\tcondition\nA01.mzML\tA1\tA\nB01.mzML\tB1\tB\n",
            Path::new(""),
            "s",
        )
        .unwrap();
        // No --mzml: the sheet is the input list.
        let r = resolve(&[], &[], &[], Some(&rows)).unwrap();
        assert_eq!(r.mzml, s(&["A01.mzML", "B01.mzML"]));
        assert_eq!(r.run_names, Some(s(&["A1", "B1"])));
        assert_eq!(r.conditions, Some(s(&["A", "B"])));
        // --mzml in another order, matched by file name; --condition wins over the column.
        let r = resolve(
            &s(&["/x/b01.mzml", "/x/A01.mzML"]),
            &[],
            &s(&["c1", "c2"]),
            Some(&rows),
        )
        .unwrap();
        assert_eq!(r.mzml, s(&["/x/b01.mzml", "/x/A01.mzML"]));
        assert_eq!(r.run_names, Some(s(&["B1", "A1"])));
        assert_eq!(r.conditions, Some(s(&["c1", "c2"])));
        // An input the sheet does not list has no condition; names must be all or none.
        let e = resolve(&s(&["A01.mzML", "C01.mzML"]), &[], &[], Some(&rows)).unwrap_err();
        assert!(e.to_string().contains("C01.mzML"), "{e}");
        let r = resolve(
            &s(&["A01.mzML", "C01.mzML"]),
            &s(&["a", "c"]),
            &[],
            Some(&rows),
        )
        .unwrap();
        assert_eq!(r.conditions, Some(s(&["A", ""])));
        // Without a sheet, the flags pass through.
        let r = resolve(&s(&["a", "b"]), &[], &[], None).unwrap();
        assert_eq!(r.run_names, None);
        assert_eq!(r.conditions, None);
        // Two rows of the same file name in different folders are ambiguous by name alone.
        let two = parse("file\nx/A01.mzML\ny/A01.mzML\n", Path::new(""), "s").unwrap();
        assert!(resolve(&s(&["/z/A01.mzML"]), &[], &[], Some(&two)).is_err());
        let r = resolve(&s(&["y/A01.mzML"]), &[], &[], Some(&two)).unwrap();
        assert_eq!(r.mzml, s(&["y/A01.mzML"]));
    }
}
