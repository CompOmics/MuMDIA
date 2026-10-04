#!/bin/bash
# Fragment-rarity score without RT: plain, + crowding, + crowding and repeat. One screen per arm
# (no --run-windows), then reduction/retention at several targets against the unfiltered run.
#   ps_crowd.sh OUTDIR NAME BIN MS2 INPUT_TABLE THREADS REF_SCORED [REF_SCORED ...]
set -u
O=$1; NAME=$2; BIN=$3; MS2=$4; IN=$5; TH=$6; shift 6
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/public/local/robbin/fs/venv/bin/python}
export MUMDIA_CACHE_DIR=off
mkdir -p "$O"
for arm in plain crowd crowd_repeat; do
  case $arm in
    plain) J='{"prescreen":{}}' ;;
    crowd) J='{"prescreen":{"crowding_exponent":0.25}}' ;;
    crowd_repeat) J='{"prescreen":{"crowding_exponent":0.25,"repeat_bonus":0.1}}' ;;
  esac
  echo "$J" > "$O/cfg_${NAME}_$arm.json"
  /usr/bin/time -v -o "$O/time_${NAME}_$arm.txt" "$BIN" --threads "$TH" prescreen \
    --ms2 "$MS2" --lib-precursors "$IN" --out "$O/${NAME}_$arm.parquet" \
    --config "$O/cfg_${NAME}_$arm.json" --write-scores > "$O/${NAME}_$arm.log" 2>&1
  echo "$(date +%T) $NAME $arm screen exit $?" >> "$O/progress.log"
  "$PY" "$HERE/ps_score_eval.py" "${NAME}_$arm" "$O/${NAME}_$arm.scores.parquet" "$IN" \
    "$O/time_${NAME}_$arm.txt" "$@" >> "$O/results.jsonl" 2>> "$O/${NAME}_$arm.eval.err"
  echo "$(date +%T) $NAME $arm eval exit $?" >> "$O/progress.log"
done
