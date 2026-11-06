//! Reading a child process's piped output, and making Python children write UTF-8.
//!
//! Every child this application starts with piped output (the engine, visDIA, uv,
//! DIA-NN) is read here, line by line, on a thread of its own. Two rules hold for all of
//! them.
//!
//! The output is read as bytes and decoded lossily. `BufRead::lines` returns an error on
//! the first line that is not valid UTF-8, and `.map_while(Result::ok)` then ended the
//! loop, which dropped the reader and closed the pipe. On Windows a Python child writes
//! a piped stdout in the ANSI code page (cp1252), so a results folder named
//! `Données_µ` printed a `0xB5` byte, the reader stopped, and the child's next write to
//! the closed pipe failed with `OSError: [Errno 22] Invalid argument` (issue #281). The
//! same applied to the engine, whose Python workers inherit its stdout and stderr.
//!
//! The reader never stops before the end of the stream. A pipe nobody drains fills up and
//! stalls the child; a pipe that is closed makes its next write fail.
//!
//! [`python_utf8`] removes the cause for Python children: with `PYTHONIOENCODING=utf-8`
//! (and `PYTHONUTF8=1`) Python writes UTF-8 to a pipe whatever the code page is.

use std::io::{BufRead, BufReader, ErrorKind, Read};
use std::process::Command;

/// Call `each` with every line of `stream`, decoded as UTF-8 with invalid bytes replaced,
/// until the stream ends. A trailing `\n` or `\r\n` is removed. A line that is not valid
/// UTF-8 is passed on with U+FFFD in place of the bad bytes, never skipped and never the
/// end of the reading. Only the end of the stream, or a read error other than an
/// interruption (the child has gone), stops it.
pub fn for_each_line_lossy<R: Read>(stream: R, mut each: impl FnMut(String)) {
    let mut reader = BufReader::new(stream);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(_) => {
                if buf.last() == Some(&b'\n') {
                    buf.pop();
                    if buf.last() == Some(&b'\r') {
                        buf.pop();
                    }
                }
                each(String::from_utf8_lossy(&buf).into_owned());
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

/// Make a Python child write its standard streams as UTF-8, whatever the console or ANSI
/// code page is. `PYTHONUTF8=1` is Python's UTF-8 mode (stdio, `open()` defaults and the
/// file-system encoding); `PYTHONIOENCODING=utf-8` covers the streams on an interpreter
/// that predates UTF-8 mode.
pub fn python_utf8(cmd: &mut Command) {
    cmd.env("PYTHONUTF8", "1").env("PYTHONIOENCODING", "utf-8");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(bytes: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        for_each_line_lossy(bytes, |l| out.push(l));
        out
    }

    #[test]
    fn a_cp1252_line_is_decoded_lossily_and_reading_goes_on() {
        // "µ" in cp1252 is the single byte 0xB5, which is not valid UTF-8.
        let mut bytes = b"mumdia-viewer: serving D:\\Donn".to_vec();
        bytes.push(0xE9); // cp1252 "é"
        bytes.extend_from_slice(b"es_");
        bytes.push(0xB5);
        bytes.extend_from_slice(b"\\run (run) at http://127.0.0.1:8050/tok/\r\n");
        bytes.extend_from_slice(b" * Serving Flask app 'x'\n");
        bytes.extend_from_slice(b"last line without newline");
        let got = lines(&bytes);
        assert_eq!(got.len(), 3, "{got:?}");
        assert!(got[0].contains('\u{FFFD}'), "{}", got[0]);
        assert!(
            got[0].ends_with("at http://127.0.0.1:8050/tok/"),
            "{}",
            got[0]
        );
        assert_eq!(got[1], " * Serving Flask app 'x'");
        assert_eq!(got[2], "last line without newline");
    }

    #[test]
    fn utf8_lines_are_unchanged() {
        let got = lines("D:\\Données_µ\\run\n\nzwei\n".as_bytes());
        assert_eq!(got, vec!["D:\\Données_µ\\run", "", "zwei"]);
    }

    #[test]
    fn a_failing_stream_ends_the_reading_without_a_panic() {
        struct Broken(bool);
        impl Read for Broken {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    return Err(std::io::Error::new(ErrorKind::BrokenPipe, "gone"));
                }
                self.0 = true;
                let line = b"one\n";
                buf[..line.len()].copy_from_slice(line);
                Ok(line.len())
            }
        }
        let mut out = Vec::new();
        for_each_line_lossy(Broken(false), |l| out.push(l));
        assert_eq!(out, vec!["one"]);
    }

    #[test]
    fn python_children_are_told_to_write_utf8() {
        let mut cmd = Command::new("never-run");
        python_utf8(&mut cmd);
        let env: Vec<(String, String)> = cmd
            .get_envs()
            .filter_map(|(k, v)| {
                Some((
                    k.to_string_lossy().into_owned(),
                    v?.to_string_lossy().into_owned(),
                ))
            })
            .collect();
        assert!(env.contains(&("PYTHONUTF8".into(), "1".into())), "{env:?}");
        assert!(
            env.contains(&("PYTHONIOENCODING".into(), "utf-8".into())),
            "{env:?}"
        );
    }
}
