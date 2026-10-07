#!/bin/bash
# Speed of the prescreen before/after a change, with score identity: OLD and NEW binaries on the
# same screen (RT windows when RUN_WINDOWS is set, else none), then score equality.
#   ps_speed.sh OUTDIR NAME OLD_BIN NEW_BIN MS2 INPUT_TABLE THREADS [RUN_WINDOWS]
set -u
O=$1; NAME=$2; OLD=$3; NEW=$4; MS2=$5; IN=$6; TH=$7; RW=${8:-}
PY=${PY:-/public/local/robbin/fs/venv/bin/python}
export MUMDIA_CACHE_DIR=off
mkdir -p "$O"
echo '{}' > "$O/cfg.json"
RWARG=(); [ -n "$RW" ] && RWARG=(--run-windows "$RW")
for which in OLD NEW; do
  bin=$OLD; [ $which = NEW ] && bin=$NEW
  /usr/bin/time -v -o "$O/time_${NAME}_$which.txt" "$bin" --threads "$TH" prescreen --ms2 "$MS2" \
    --lib-precursors "$IN" "${RWARG[@]}" --out "$O/${NAME}_$which.parquet" --config "$O/cfg.json" \
    --write-scores > "$O/${NAME}_$which.log" 2>&1
  echo "$(date +%T) $NAME $which exit $? $(grep -E 'Elapsed' "$O/time_${NAME}_$which.txt" | rev | cut -d' ' -f1 | rev) $(grep -E 'Maximum res' "$O/time_${NAME}_$which.txt" | awk '{printf "%.1fGB", $6/1048576}')" >> "$O/progress.log"
done
"$PY" - "$O/${NAME}_OLD.scores.parquet" "$O/${NAME}_NEW.scores.parquet" >> "$O/progress.log" 2>&1 <<'EOF'
import sys, numpy as np, pyarrow.parquet as pq
a = pq.read_table(sys.argv[1], columns=["candidate_id", "score"]).to_pandas()
b = pq.read_table(sys.argv[2], columns=["candidate_id", "score"]).to_pandas()
same_ids = (a.candidate_id.values == b.candidate_id.values).all()
x, y = a.score.to_numpy(), b.score.to_numpy()
eq = np.array_equal(np.nan_to_num(x, nan=-1), np.nan_to_num(y, nan=-1))
print(f"scores identical: {eq} (ids aligned {same_ids}, n {len(x)}, max abs diff {np.nanmax(np.abs(x - y)):.3g})")
EOF
