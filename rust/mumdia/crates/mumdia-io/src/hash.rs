//! blake3 content hashing (PLAN.md Section 3.3: every artifact records a
//! content hash; config is hashed to invalidate downstream artifacts).

use anyhow::{Context, Result};
use std::io::Read;

/// blake3 hash of a file's bytes, hex-encoded.
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
