#!/bin/bash
# A FASTA search with many variable modifications, with and without the prescreen before
# prediction, against the standard modifications: the configurations the desktop Search tab
# derives. Arms run one after another for clean timing, then two more NN seeds per arm (the
# run itself is seed 0).
#   ps_mods.sh ROOT BIN MZML FASTA CFG THREADS
# Arms: OFF (CFG as given), MODS (CFG + the modification set below), MODSPRE (MODS + the
# fragment-rarity score before prediction with crowding 0.25, balanced).
set -u
ROOT=$1; BIN=$2; MZ=$3; FASTA=$4; CFG=$5; THREADS=$6
ARMS=${ARMS:-"OFF MODS MODSPRE"}
export MUMDIA_CACHE_DIR=off
mkdir -p "$ROOT"
log() { echo "$(date +%T) $*" >> "$ROOT/progress.log"; }
python3 - "$CFG" "$ROOT" <<'EOF'
import json, sys
base = json.load(open(sys.argv[1]))
mods = {
    "fixed_mods": [{"residue": "C", "name": "Carbamidomethyl"}],
    "variable_mods": [{"residue": r, "name": n} for n, r in [
        ("Oxidation", "M"), ("Phospho", "S"), ("Phospho", "T"), ("Phospho", "Y"),
        ("Acetyl", "K"), ("Deamidated", "N"), ("Deamidated", "Q")]],
    "max_variable_mods": 1,
}
for arm in ("OFF", "MODS", "MODSPRE"):
    c = json.loads(json.dumps(base))
    if arm != "OFF":
        c.setdefault("peptidoforms", {}).update(mods)
    if arm == "MODSPRE":
        c["prescreen"] = {"score_before_prediction": True, "crowding_exponent": 0.25,
                          "preset": "balanced"}
    json.dump(c, open(f"{sys.argv[2]}/cfg_{arm}.json", "w"), indent=2)
EOF
log "START bin=$(sha256sum "$BIN" | cut -c1-16) mz=$(basename "$MZ") fasta=$(basename "$FASTA")"
for arm in $ARMS; do
  /usr/bin/time -v -o "$ROOT/time_$arm.txt" "$BIN" --threads "$THREADS" run --fasta "$FASTA" \
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
