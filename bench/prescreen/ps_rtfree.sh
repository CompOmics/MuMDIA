#!/bin/bash
# End-to-end check of prescreen.placement = before_prediction: a plain `mumdia run` (OFF) and the
# same run with the prescreen before prediction (PRE), one after the other for clean timing,
# then two more NN seeds per arm (the run itself is seed 0).
#   ps_rtfree.sh ROOT BIN MODE MZML A B CFG THREADS
#   MODE fasta: A = FASTA, B unused;  MODE lib: A = lib precursors, B = lib fragments
set -u
ROOT=$1; BIN=$2; MODE=$3; MZ=$4; A=$5; B=$6; CFG=$7; THREADS=$8
PRESET=${PRESET:-balanced}
export MUMDIA_CACHE_DIR=off
mkdir -p "$ROOT"
log() { echo "$(date +%T) $*" >> "$ROOT/progress.log"; }
python3 - "$CFG" "$ROOT/cfg_pre.json" "$PRESET" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c["prescreen"] = {"enabled": True, "placement": "before_prediction", "preset": sys.argv[3],
                  "write_scores": True}
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
if [ "$MODE" = fasta ]; then IN=(--fasta "$A"); else IN=(--lib-precursors "$A" --lib-fragments "$B"); fi
log "START bin=$(sha256sum "$BIN" | cut -c1-16) mode=$MODE mz=$(basename "$MZ")"
for arm in OFF PRE; do
  c=$CFG; [ $arm = PRE ] && c=$ROOT/cfg_pre.json
  /usr/bin/time -v -o "$ROOT/time_$arm.txt" "$BIN" --threads "$THREADS" run "${IN[@]}" --mzml "$MZ" \
    --out-dir "$ROOT/$arm" --config "$c" > "$ROOT/$arm.log" 2>&1
  log "$arm exit $?"
done
for arm in OFF PRE; do
  c=$CFG; [ $arm = PRE ] && c=$ROOT/cfg_pre.json
  for s in 1 2; do
    mkdir -p "$ROOT/$arm/s$s"
    ( export MUMDIA_NN_SEED=$s; "$BIN" --threads 16 rescore --competed "$ROOT/$arm/psms_competed.parquet" \
        --out "$ROOT/$arm/s$s/psms_scored.parquet" --work-dir "$ROOT/$arm/s$s/work" --config "$c" \
        > "$ROOT/${arm}_rescore_s$s.log" 2>&1; log "${arm}_rescore_s$s exit $?"; rm -rf "$ROOT/$arm/s$s/work" ) &
  done
done
wait
log "ALL_DONE"
touch "$ROOT/ALL_DONE"
