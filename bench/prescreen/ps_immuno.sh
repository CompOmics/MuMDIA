#!/bin/bash
# Prescreen on a large library with an existing calibrated run as the baseline (docs/32 immuno
# 8-12-mer case). Phase 1, strictly sequential for clean wall/RSS: one prescreen (balanced, with
# scores), other presets derived from the scores, then extract -> features -> compete per arm.
# Phase 2: NN rescores, one arm's seeds at a time.
#   ps_immuno.sh ROOT BIN BASE_RUN LIBP LIBF CFG THREADS
set -u
ROOT=$1; BIN=$2; B=$3; LIB=$4; LIBF=$5; CFG=$6; THREADS=$7
ARMS=${ARMS:-"OFF BAL STR"}
SEEDS=${SEEDS:-"0 1 2"}
PY=${PY:-/public/local/robbin/fs/venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export MUMDIA_CACHE_DIR=off
S=$B/spectra
mkdir -p "$ROOT"
log() { echo "$(date +%T) $*" >> "$ROOT/progress.log"; }
T() {
  local name=$1; shift
  /usr/bin/time -v -o "$ROOT/time_$name.txt" "$@" > "$ROOT/$name.log" 2>&1
  local rc=$?; log "$name exit $rc"; return $rc
}
log "START bin=$(sha256sum "$BIN" | cut -c1-16) base=$B lib=$LIB"
mkdir -p "$ROOT/BAL"
T BAL_screen "$BIN" --threads "$THREADS" prescreen --ms2 "$S/spectra_ms2.parquet" \
  --lib-precursors "$LIB" --run-windows "$B/run_windows.parquet" \
  --out "$ROOT/BAL/survivors.parquet" --config "$CFG" --preset balanced --write-scores || exit 1
for p in "SEN 0.52" "STR 0.75" "AGG 0.90"; do
  set -- $p; mkdir -p "$ROOT/$1"
  "$PY" "$HERE/ps_derive.py" "$ROOT/BAL/survivors.scores.parquet" "$2" "$ROOT/$1/survivors.parquet" >> "$ROOT/progress.log" 2>&1
done
for arm in $ARMS; do
  A=$ROOT/$arm; mkdir -p "$A"; R=""
  [ "$arm" != OFF ] && R="--restrict-candidates $A/survivors.parquet"
  T "${arm}_extract" "$BIN" --threads "$THREADS" extract --ms2 "$S/spectra_ms2.parquet" \
    --ms1 "$S/spectra_ms1.parquet" --lib-precursors "$LIB" --lib-fragments "$LIBF" \
    --run-windows "$B/run_windows.parquet" --mass-cal "$B/seed_psms.parquet.masscal.json" \
    --out-psms "$A/psms_extracted.parquet" --out-chromatograms "$A/chromatograms.parquet" \
    $R --config "$CFG" || continue
  T "${arm}_features" "$BIN" --threads "$THREADS" features --psms-extracted "$A/psms_extracted.parquet" \
    --chromatograms "$A/chromatograms.parquet" --seed-psms "$B/seed_psms.parquet" \
    --out "$A/features.parquet" --out-pin "$A/run.pin" --config "$CFG" || continue
  rm -f "$A/chromatograms.parquet"
  T "${arm}_compete" "$BIN" --threads "$THREADS" compete --features "$A/features.parquet" \
    --out "$A/psms_competed.parquet" --config "$CFG" || continue
  rm -f "$A/features.parquet"
done
log "PHASE1_DONE"
for arm in $ARMS; do
  A=$ROOT/$arm
  [ -f "$A/psms_competed.parquet" ] || continue
  for s in $SEEDS; do
    mkdir -p "$A/s$s"
    ( export MUMDIA_NN_SEED=$s; T "${arm}_rescore_s$s" "$BIN" --threads 32 rescore \
        --competed "$A/psms_competed.parquet" --out "$A/s$s/psms_scored.parquet" \
        --work-dir "$A/s$s/work" --config "$CFG"; rm -rf "$A/s$s/work" ) &
  done
  wait
done
log "ALL_DONE"
touch "$ROOT/ALL_DONE"
