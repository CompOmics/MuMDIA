#!/bin/bash
# Decoy test for the predictor comparison under entrapment: the targets of each
# ps_entrap_pred.sh library get reverse decoys whose intensities are copied ion-for-ion from
# their target (make_reverse_decoys.py, the DIA-NN-path recipe) instead of being predicted by
# the same model, then each library is searched on the entrapment run with two more NN seeds.
# If AlphaPeptDeep's elevated entrapment FDP comes from predicting reversed sequences, it
# returns to the MS2PIP level here; the MS2PIP arm is the control for the decoy builder.
#   ps_entrap_decoy.sh ROOT PRED_ROOT BIN MZML THREADS
set -u
ROOT=$1; PRED=$2; BIN=$3; MZ=$4; THREADS=$5
ARMS=${ARMS:-"MS2PIP PEPTDEEP"}
PY=${PY:-/public/local/robbin/fs/venv/bin/python}
SCRIPTS=${SCRIPTS:-/public/local/robbin/ps/src13/scripts}
export MUMDIA_CACHE_DIR=off
mkdir -p "$ROOT"
log() { echo "$(date +%T) $*" >> "$ROOT/progress.log"; }
log "START bin=$(sha256sum "$BIN" | cut -c1-16) pred=$PRED"
for arm in $ARMS; do
  mkdir -p "$ROOT/lib_$arm"
  "$PY" "$SCRIPTS/make_reverse_decoys.py" "$PRED/lib_$arm/lib_precursors.parquet" \
    "$PRED/lib_$arm/lib_fragments.parquet" "$ROOT/lib_$arm/lib_precursors.parquet" \
    "$ROOT/lib_$arm/lib_fragments.parquet" > "$ROOT/lib_$arm.log" 2>&1
  log "lib_$arm exit $?"
  cp "$PRED/cfg_$arm.json" "$ROOT/cfg_$arm.json"
done
for arm in $ARMS; do
  /usr/bin/time -v -o "$ROOT/time_$arm.txt" "$BIN" --threads "$THREADS" run \
    --lib-precursors "$ROOT/lib_$arm/lib_precursors.parquet" \
    --lib-fragments "$ROOT/lib_$arm/lib_fragments.parquet" \
    --mzml "$MZ" --out-dir "$ROOT/$arm" --config "$ROOT/cfg_$arm.json" > "$ROOT/$arm.log" 2>&1
  log "$arm exit $?"
done
for arm in $ARMS; do
  for s in 1 2; do
    mkdir -p "$ROOT/$arm/s$s"
    ( export MUMDIA_NN_SEED=$s; "$BIN" --threads 16 rescore --competed "$ROOT/$arm/psms_competed.parquet" \
        --out "$ROOT/$arm/s$s/psms_scored.parquet" --work-dir "$ROOT/$arm/s$s/work" \
        --config "$ROOT/cfg_$arm.json" > "$ROOT/${arm}_rescore_s$s.log" 2>&1
      log "${arm}_rescore_s$s exit $?"; rm -rf "$ROOT/$arm/s$s/work" ) &
  done
done
wait
log "ALL_DONE"
touch "$ROOT/ALL_DONE"
