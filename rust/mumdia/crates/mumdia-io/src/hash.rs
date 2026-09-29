//! blake3 content hashing (docs/03_io_layer.md): every artifact records a
//! content hash; config is hashed to invalidate downstream artifacts.

use anyhow::{Context, Result};
use std::io::{Read, Write};

/// blake3 hash of a file's bytes, hex-encoded.
///
/// This always reads the file. It is deliberately not memoised: `features` uses it in its
/// tests as an independent integrity check of a published artifact, and a cache keyed on
/// the path would answer that check from the writer's own claim. A writer that hashes its
/// bytes as it writes them ([`HashingWrite`], `TableWriter::with_content_hash` and the
/// other `*_hashed` writers in `table.rs`) removes the read-back instead.
pub fn blake3_file(path: &str) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("hashing {path}"))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// blake3 hash of a string (used for config hashing).
pub fn blake3_str(s: &str) -> String {
    blake3::hash(s.as_bytes()).to_hex().to_string()
}

/// A writer that forwards every byte to `inner` and, when asked to, feeds the same bytes to
/// a blake3 hasher on the way.
///
/// Every artifact used to be hashed by reading it back after it was published
/// ([`blake3_file`]), so each byte crossed the disk or the page cache twice. The parquet
/// writers in this crate only ever append to their sink (they never seek), so the bytes a
/// sink sees, in order, are exactly the bytes of the finished file, and the digest of the
/// stream is the digest of the file. [`HashingWrite::finish`] returns it.
///
/// Only the bytes `inner` accepted are hashed (`write` may be partial), so a short write
/// cannot desynchronise the digest from the file. With hashing off this is a plain
/// pass-through that costs one branch per call.
pub struct HashingWrite<W: Write> {
    inner: W,
    hasher: Option<blake3::Hasher>,
    written: u64,
}

impl<W: Write> HashingWrite<W> {
    /// Wrap `inner`; `hash` turns the digest on.
    pub fn new(inner: W, hash: bool) -> HashingWrite<W> {
        HashingWrite {
            inner,
            hasher: hash.then(blake3::Hasher::new),
            written: 0,
        }
    }

    /// Bytes forwarded to `inner` so far.
    pub fn bytes_written(&self) -> u64 {
        self.written
    }

    /// The inner writer and the hex digest of every byte it accepted, or `None` when the
    /// digest was not asked for. The caller flushes any buffer in front of this first.
    pub fn finish(self) -> (W, Option<String>) {
        let digest = self.hasher.map(|h| h.finalize().to_hex().to_string());
        (self.inner, digest)
    }
}

impl<W: Write> Write for HashingWrite<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        if let Some(h) = self.hasher.as_mut() {
            h.update(&buf[..n]);
        }
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sink that accepts at most `cap` bytes per call, to exercise partial writes.
    struct Trickle {
        out: Vec<u8>,
        cap: usize,
    }

    impl Write for Trickle {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let n = buf.len().min(self.cap);
            self.out.extend_from_slice(&buf[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn the_streamed_digest_is_the_digest_of_the_bytes_accepted() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 31 % 251) as u8).collect();
        let mut w = HashingWrite::new(
            Trickle {
                out: Vec::new(),
                cap: 7_777,
            },
            true,
        );
        // write_all loops over the partial writes; every accepted byte is hashed once.
        w.write_all(&data[..123]).unwrap();
        w.write_all(&data[123..]).unwrap();
        assert_eq!(w.bytes_written(), data.len() as u64);
        let (sink, digest) = w.finish();
        assert_eq!(sink.out, data);
        assert_eq!(digest.unwrap(), blake3::hash(&data).to_hex().to_string());

        let mut off = HashingWrite::new(Vec::new(), false);
        off.write_all(&data).unwrap();
        let (bytes, none) = off.finish();
        assert_eq!(bytes, data);
        assert!(none.is_none());

        let empty = HashingWrite::new(Vec::new(), true);
        assert_eq!(
            empty.finish().1.unwrap(),
            blake3::hash(b"").to_hex().to_string()
        );
    }
}
