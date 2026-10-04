#!/bin/bash
# Standalone RT-free tag prefilter on one dataset: `mumdia prescreen --tags-only` without run
# windows, timed, then retention against the unfiltered run's accepted precursors.
#   ps_tagpre.sh OUTDIR NAME BIN MS2 INPUT_TABLE THREADS REF_SCORED [REF_SCORED ...]
set -u
O=$1; NAME=$2; BIN=$3; MS2=$4; IN=$5; TH=$6; shift 6
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/public/local/robbin/fs/venv/bin/python}
export MUMDIA_CACHE_DIR=off
mkdir -p "$O"
echo '{}' > "$O/cfg_default.json"
/usr/bin/time -v -o "$O/time_$NAME.txt" "$BIN" --threads "$TH" prescreen --tags-only \
  --ms2 "$MS2" --lib-precursors "$IN" --out "$O/$NAME.survivors.parquet" \
  --config "$O/cfg_default.json" > "$O/$NAME.log" 2>&1
echo "$(date +%T) $NAME screen exit $?" >> "$O/progress.log"
"$PY" "$HERE/ps_tagpre_eval.py" "$NAME" "$O/$NAME.survivors.parquet" "$IN" "$O/time_$NAME.txt" "$@" \
  >> "$O/results.jsonl" 2>> "$O/$NAME.eval.err"
echo "$(date +%T) $NAME eval exit $?" >> "$O/progress.log"
