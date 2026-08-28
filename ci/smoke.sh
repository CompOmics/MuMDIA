#!/usr/bin/env bash
# End-to-end smoke test on the generated fixture. Run by CI, and runnable locally.
#
#   ci/smoke.sh [work_dir]
#
# Needs: a built `mumdia` binary (found automatically, or set MUMDIA_BIN) and a
# Python with pyarrow. No sidecar, no network, no data file in the repository: the
# fixture is generated from `test_data/fixture.fasta` and from the library the
# engine itself builds out of it, so the planted peaks cannot disagree with the
# engine's mass model.
#
# What it covers that the Rust suite does not: mzML parsing (`convert`), the
# `digest -> peptidoforms -> predict-frag` library build, the `run` orchestrator
# and its manifest, RT calibration on real anchors, and `quant` and `report`
# writing files. The Rust integration test builds its inputs in process and starts
# at `extract`, so none of that was exercised before.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
work="${1:-${TMPDIR:-/tmp}/mumdia_smoke}"
cfg="configs/examples/native.json"

# Locate the release binary.
#
# The CARGO-REPORTED target directory comes first, because a redirected
# `build.target-dir` -- which `.cargo/config.toml.example` documents, to keep build
# artifacts out of a synced folder -- means `rust/mumdia/target/` can still hold a
# months-old binary from before the redirect. Not hypothetical: that is exactly how a
# writer/reader mismatch (convert moved to LargeList, spectra.rs still downcast to
# List) passed here and failed in CI. An in-tree path that shadows the real build
# output is worse than no path at all.
find_bin() {
    if [ -n "${MUMDIA_BIN:-}" ]; then echo "$MUMDIA_BIN"; return; fi
    target=""
    if command -v cargo > /dev/null 2>&1; then
        # Run cargo from INSIDE the workspace, not with --manifest-path from the repo
        # root: cargo discovers `.cargo/config.toml` by walking up from the CURRENT
        # DIRECTORY, not from the manifest, so from the root it reports the in-tree
        # `target/` and misses the redirect entirely. That is the bug this whole
        # function exists to avoid, so getting it wrong here defeats the point.
        target=$( (cd rust/mumdia && cargo metadata --format-version 1 --no-deps 2>/dev/null) \
                 | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])' \
                 2>/dev/null || true)
    fi
    for cand in \
        "${target:+$target/release/mumdia}" \
        "${target:+$target/release/mumdia.exe}" \
        rust/mumdia/target/release/mumdia \
        rust/mumdia/target/release/mumdia.exe
    do
        [ -n "$cand" ] && [ -x "$cand" ] && { echo "$cand"; return; }
    done
    command -v mumdia || { echo "no mumdia binary found; set MUMDIA_BIN" >&2; exit 2; }
}

# Before find_bin, which shells out to it.
PY="${PYTHON:-python}"
BIN="$(find_bin)"
echo "=== smoke: binary $BIN"
"$BIN" --version
rm -rf "$work"
mkdir -p "$work"

# The engine must be able to say the configuration is runnable before we run it.
echo "=== smoke: doctor"
"$BIN" doctor --config "$cfg"

# 1. Library, standalone, so the fixture generator can read the exact masses the
#    engine will look for.
echo "=== smoke: build the fixture library"
"$BIN" digest --fasta test_data/fixture.fasta --out "$work/peptides.parquet" --config "$cfg"
"$BIN" peptidoforms --peptides "$work/peptides.parquet" --out "$work/peptidoforms.parquet" \
    --config "$cfg"
"$BIN" predict-frag --peptidoforms "$work/peptidoforms.parquet" \
    --out-precursors "$work/lib_prec.parquet" --out-fragments "$work/lib_frag.parquet" \
    --work-dir "$work/sidecar" --config "$cfg"

# 2. The mzML.
echo "=== smoke: generate the fixture mzML"
# --quiet rather than `| head -3`: truncating the generator's output closes its
# stdout early, and on a Windows console that surfaces as
# `OSError: [Errno 22] Invalid argument` during interpreter shutdown rather than as
# EPIPE, so the script exited 120 with no failing assertion to point at. It passed
# on Linux and locally and failed only on the CI Windows runner.
"$PY" ci/make_fixture_mzml.py \
    --precursors "$work/lib_prec.parquet" --fragments "$work/lib_frag.parquet" \
    --out "$work/fixture.mzML" --manifest "$work/planted.json" \
    --n-planted 160 --windows 8 --quiet

# 3. The full single-run orchestrator, from the FASTA, so digest runs inside `run`
#    as well and the library it builds is compared against the one the fixture was
#    generated from by construction.
echo "=== smoke: run"
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out" --config "$cfg" --threads 2

# 4. A second run into a fresh directory, to assert byte-level determinism.
echo "=== smoke: run again for determinism"
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out2" --config "$cfg" --threads 2 > "$work/run2.log" 2>&1 \
    || { tail -20 "$work/run2.log"; exit 1; }

# 5. Assertions.
echo "=== smoke: assertions"
"$PY" ci/check_smoke.py --out-dir "$work/out" --planted "$work/planted.json" \
    --compare-peptides "$work/out2/peptides.tsv"

echo "=== smoke: also check inspect and report run standalone"
"$BIN" inspect "$work/out/psms_scored.parquet" > /dev/null
"$BIN" report --psms-scored "$work/out/psms_scored.parquet" --out-dir "$work/report_only" \
    --peptide-quant "$work/out/peptide_quant.parquet" --q 0.05 > /dev/null
test -s "$work/report_only/peptides.tsv"

# Hashes of the user-facing outputs, for the cross-platform comparison in CI.
# Measured 2026-08-27: these are IDENTICAL on Windows and Linux, down to the
# quantity digits, so the native pipeline is byte-reproducible across operating
# systems and not merely across runs on one machine. That is a property worth
# keeping, and the only way to keep it is to check it.
#
# No expected value is committed. A golden hash would have to be updated by every
# legitimate change to scoring, which turns an improvement into a chore and
# eventually into a rubber stamp. CI compares the two platforms against each
# other instead.
hashfile="$repo/smoke_output_hashes.txt"
: > "$hashfile"
for f in peptides.tsv proteins.tsv; do
    if command -v sha256sum >/dev/null 2>&1; then
        h=$(sha256sum "$work/out/$f" | cut -d" " -f1)
    else
        h=$(shasum -a 256 "$work/out/$f" | cut -d" " -f1)
    fi
    echo "$h  $f" >> "$hashfile"
done
echo "=== smoke: output hashes"
cat "$hashfile"

echo "SMOKE_OK"
