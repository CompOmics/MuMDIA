#!/usr/bin/env bash
# The advisory scan of both lockfiles, with its one list of deliberate ignores.
#
#   ci/cargo_audit.sh        (from the repository root; cargo-audit on PATH)
#
# Run by ci.yml's `cargo audit` job and by release.yml's tag gate, so a release cannot
# ship an advisory nobody handled even when the scan was advisory on the pull request.
#
# Vulnerabilities fail. Unmaintained-crate warnings do not: `paste` is a compile-time
# proc macro whose 1.0.15 is its final release, so there is nothing to bump to and a
# hard failure would only train people to ignore this scan. Deny the class that has a
# fix.
set -euo pipefail
cd "$(dirname "$0")/.."

# Thirteen zstd advisories are ignored deliberately, all on zstd 0.13.3 /
# zstd-safe 7.3.0, which only timsrust 0.4.2 (the diaPASEF reader) pulls in:
# RUSTSEC-2026-0391 (data of an abandoned stream on a reused context), -0398 and
# -0396 (Seekable), -0403 (failed streaming decode of a legacy frame), and the
# unsound -0395, -0397, -0399, -0400, -0401, -0402, -0404, -0405, -0406
# (dictionaries, cloned or multithreaded contexts, OutBuffer/InBuffer positions,
# dictionary training). timsrust calls only `zstd::decode_all`, a fresh one-shot
# decoder per TDF frame, with no dictionary, Seekable, cloning, multithreading or
# context reuse. parquet's zstd is on the fixed 0.14.3 / zstd-safe 8.2.0. The fix
# for the timsrust path is zstd >= 0.14.3, which needs timsrust >= 0.6; drop these
# ignores when the reader moves there.
#
# The zip advisories of nethost-sys need no ignore: the vendored nethost-sys
# (rust/mumdia/vendor/nethost-sys, its `resolves` list) uses zip >= 9.0.5.
echo "--- rust/mumdia/Cargo.lock ---"
(cd rust/mumdia && cargo audit --deny unsound --deny yanked \
  --ignore RUSTSEC-2026-0391 --ignore RUSTSEC-2026-0395 --ignore RUSTSEC-2026-0396 \
  --ignore RUSTSEC-2026-0397 --ignore RUSTSEC-2026-0398 --ignore RUSTSEC-2026-0399 \
  --ignore RUSTSEC-2026-0400 --ignore RUSTSEC-2026-0401 --ignore RUSTSEC-2026-0402 \
  --ignore RUSTSEC-2026-0403 --ignore RUSTSEC-2026-0404 --ignore RUSTSEC-2026-0405 \
  --ignore RUSTSEC-2026-0406)

# The desktop application has its own lockfile (docs/29, work package C). One
# advisory is ignored deliberately: RUSTSEC-2024-0429, an unsound iterator in glib
# 0.18's `VariantStrIter`. Tauri 2's Linux toolkit is gtk 0.18, which pins glib 0.18;
# the fix is glib 0.20; and the application never touches a GVariant. Drop the ignore
# when Tauri moves to gtk4 / glib 0.20.
echo "--- desktop/Cargo.lock ---"
cargo audit --file desktop/Cargo.lock --deny unsound --deny yanked \
  --ignore RUSTSEC-2024-0429
