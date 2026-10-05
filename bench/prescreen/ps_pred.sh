#!/bin/bash
# Fragment-intensity predictor comparison on one file, same digest for every arm: MS2PIP and
# AlphaPeptDeep from the FASTA, and the DIA-NN-predicted library for the same FASTA and
# digest. RT is DeepLC (with the default multi-head calibration) in every arm. Arms run one
# after another for clean timing, then two more NN seeds per arm (the run is seed 0).
#   ps_pred.sh ROOT BIN MZML FASTA DIANN_PREC DIANN_FRAG CFG THREADS
set -u
ROOT=$1; BIN=$2; MZ=$3; FASTA=$4; DP=$5; DF=$6; CFG=$7; THREADS=$8
ARMS=${ARMS:-"MS2PIP PEPTDEEP DIANN"}
PEPTDEEP_PY=${PEPTDEEP_PY:-/public/local/robbin/fs/venv_peptdeep/bin/python}
export MUMDIA_CACHE_DIR=off
mkdir -p "$ROOT"
log() { echo "$(date +%T) $*" >> "$ROOT/progress.log"; }
python3 - "$CFG" "$ROOT" "$PEPTDEEP_PY" <<'EOF'
import json, sys
base = json.load(open(sys.argv[1]))
base.setdefault("digest", {}).update({"missed_cleavages": 1, "min_len": 7, "max_len": 30})
base.setdefault("peptidoforms", {}).update({"charge_min": 2, "charge_max": 3})
for arm in ("MS2PIP", "PEPTDEEP", "DIANN"):
    c = json.loads(json.dumps(base))
    if arm == "PEPTDEEP":
        c["predict_frag"].update({"predictor": "peptdeep", "peptdeep_python": sys.argv[3],
                                  "peptdeep_model": "generic", "peptdeep_nce": 30.0,
                                  "peptdeep_instrument": "Lumos"})
    json.dump(c, open(f"{sys.argv[2]}/cfg_{arm}.json", "w"), indent=2)
EOF
log "START bin=$(sha256sum "$BIN" | cut -c1-16) mz=$(basename "$MZ") fasta=$(basename "$FASTA")"
for arm in $ARMS; do
  if [ "$arm" = DIANN ]; then IN=(--lib-precursors "$DP" --lib-fragments "$DF"); else IN=(--fasta "$FASTA"); fi
  /usr/bin/time -v -o "$ROOT/time_$arm.txt" "$BIN" --threads "$THREADS" run "${IN[@]}" \
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
