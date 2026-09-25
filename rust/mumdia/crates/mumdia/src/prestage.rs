//! Work the orchestrators do before their first stage: hashing the inputs for the
//! manifest off the critical path.
//!
//! `run` and `run-experiment` record a blake3 hash of every input file (docs/29 #15).
//! The hash used to be taken serially, one input after another, before the first stage
//! started. On a large experiment that is minutes of wall time with nothing else running:
//! `blake3_file` is a single-threaded sequential read, and seven mzML files plus a ~30 GB
//! fragment table at the array's cold read rate is about four minutes for the fragments
//! alone (perf survey critic item 4). [`InputHashes`] takes the same hashes on one
//! background thread while the pipeline starts, and the orchestrator joins it only when it
//! writes the manifest, so the recorded values are the ones the serial loop produced.

use std::collections::BTreeMap;
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::Result;
use mumdia_core::manifest::Manifest;
use tracing::{info, warn};

/// One hashed input: its manifest role, its path, and `(bytes, blake3)` or why not.
struct Hashed {
    role: String,
    path: String,
    result: Result<(u64, String)>,
}

fn hash_all(what: &str, inputs: Vec<(String, String)>) -> Vec<Hashed> {
    let t0 = Instant::now();
    let mut bytes_total = 0u64;
    let out: Vec<Hashed> = inputs
        .into_iter()
        .map(|(role, path)| {
            let result = std::fs::metadata(&path)
                .map_err(anyhow::Error::from)
                .and_then(|m| Ok((m.len(), mumdia_io::hash::blake3_file(&path)?)));
            if let Ok((b, _)) = &result {
                bytes_total += b;
            }
            Hashed { role, path, result }
        })
        .collect();
    let secs = t0.elapsed().as_secs_f64();
    info!(
        inputs = out.len(),
        bytes = bytes_total,
        elapsed_ms = t0.elapsed().as_millis(),
        mb_per_s = if secs > 0.0 {
            bytes_total as f64 / 1e6 / secs
        } else {
            0.0
        },
        "{what}: inputs hashed for the manifest (background thread)"
    );
    out
}

/// Input hashes being taken on a background thread.
///
/// The thread reads each input once, sequentially, in the order given, which the callers
/// choose to match the order the pipeline reads them, so the two reads of one file are
/// close together and the second tends to come from the page cache. It is joined by
/// [`InputHashes::record`], which records exactly what the old serial loop recorded: the
/// same roles, paths, byte counts and hashes, in a `BTreeMap` keyed by role, so neither the
/// thread's timing nor the order of the list reaches the manifest.
pub struct InputHashes {
    what: &'static str,
    state: State,
}

enum State {
    Running(JoinHandle<Vec<Hashed>>),
    /// The thread could not be spawned, so the hashes were taken inline.
    Done(Vec<Hashed>),
}

impl InputHashes {
    /// Start hashing `(role, path)` pairs. `what` names the orchestrator in the log.
    pub fn spawn(what: &'static str, inputs: Vec<(String, String)>) -> Self {
        let for_thread = inputs.clone();
        let state = match std::thread::Builder::new()
            .name("mumdia-input-hash".into())
            .spawn(move || hash_all(what, for_thread))
        {
            Ok(h) => State::Running(h),
            Err(e) => {
                warn!(error = %e, "{what}: cannot start the input-hash thread; hashing inline");
                State::Done(hash_all(what, inputs))
            }
        };
        Self { what, state }
    }

    /// Wait for the hashes, record each in `man`, and return the hash of every input that
    /// hashed, keyed by role, for a caller that records the same file as an artifact.
    ///
    /// An input that could not be read is left out of the manifest with a warning, as the
    /// serial loop did: a missing input is a preflight error, so one that became unreadable
    /// since then is better reported than turned into a failed run at the very end.
    pub fn record(self, man: &mut Manifest) -> BTreeMap<String, String> {
        let t_wait = Instant::now();
        let hashed = match self.state {
            State::Running(h) => match h.join() {
                Ok(v) => v,
                Err(_) => {
                    warn!(
                        "{}: the input-hash thread panicked; the manifest records no input \
                         hashes",
                        self.what
                    );
                    Vec::new()
                }
            },
            State::Done(v) => v,
        };
        let waited = t_wait.elapsed().as_millis();
        if waited > 1000 {
            // The pipeline finished before the hashes did, which only happens on a short
            // run over large inputs; say so, because it is time the run spent waiting.
            info!(
                waited_ms = waited,
                "{}: waited for the input hashes before writing the manifest", self.what
            );
        }
        let mut by_role = BTreeMap::new();
        for h in hashed {
            match h.result {
                Ok((bytes, hash)) => {
                    man.record_input(&h.role, &h.path, bytes, hash.clone());
                    by_role.insert(h.role, hash);
                }
                Err(e) => warn!(
                    role = %h.role,
                    path = %h.path,
                    error = %format!("{e:#}"),
                    "{}: could not hash input for the manifest",
                    self.what
                ),
            }
        }
        by_role
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_hashes_equal_the_serial_ones_and_skip_unreadable_inputs() {
        let dir = std::env::temp_dir().join(format!("mumdia_prestage_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.bin");
        let b = dir.join("b.bin");
        std::fs::write(&a, vec![7u8; 200_000]).unwrap();
        std::fs::write(&b, b"small").unwrap();
        let a = a.to_str().unwrap().to_string();
        let b = b.to_str().unwrap().to_string();
        let missing = dir.join("missing.bin").to_str().unwrap().to_string();

        let mut serial = Manifest::new("{}".into(), "c".repeat(64));
        for (role, path) in [("mzml", &a), ("fasta", &b)] {
            let bytes = std::fs::metadata(path).unwrap().len();
            serial.record_input(
                role,
                path,
                bytes,
                mumdia_io::hash::blake3_file(path).unwrap(),
            );
        }

        let mut bg = Manifest::new("{}".into(), "c".repeat(64));
        // A different order and an unreadable input: neither reaches the manifest.
        let hashes = InputHashes::spawn(
            "test",
            vec![
                ("fasta".into(), b.clone()),
                ("lib_precursors".into(), missing),
                ("mzml".into(), a.clone()),
            ],
        );
        let by_role = hashes.record(&mut bg);
        assert_eq!(
            serde_json::to_string(&serial.inputs).unwrap(),
            serde_json::to_string(&bg.inputs).unwrap()
        );
        assert_eq!(by_role.len(), 2);
        assert_eq!(by_role["mzml"], serial.inputs["mzml"].content_hash);
    }
}
