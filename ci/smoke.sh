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

find_bin() {
    if [ -n "${MUMDIA_BIN:-}" ]; then echo "$MUMDIA_BIN"; return; fi
    for cand in \
        rust/mumdia/target/release/mumdia \
        rust/mumdia/target/release/mumdia.exe \
        "$HOME/mumdia_build/release/mumdia" \
        "C:/Users/robbi/mumdia_build/release/mumdia.exe"
    do
        [ -x "$cand" ] && { echo "$cand"; return; }
    done
    command -v mumdia || { echo "no mumdia binary found; set MUMDIA_BIN" >&2; exit 2; }
}

BIN="$(find_bin)"
PY="${PYTHON:-python}"
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
"$PY" ci/make_fixture_mzml.py \
    --precursors "$work/lib_prec.parquet" --fragments "$work/lib_frag.parquet" \
    --out "$work/fixture.mzML" --manifest "$work/planted.json" \
    --n-planted 160 --windows 8 | head -3

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
"$BIN" report --scored "$work/out/psms_scored.parquet" --out-dir "$work/report_only" \
    --peptide-quant "$work/out/peptide_quant.parquet" --q 0.05 > /dev/null
test -s "$work/report_only/peptides.tsv"

echo "SMOKE_OK"
