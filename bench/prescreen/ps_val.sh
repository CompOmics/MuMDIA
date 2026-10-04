#!/bin/bash
# Prescreen validation on one acquisition: a baseline `mumdia run`, then a paired chain
# (extract -> features -> compete -> rescore x seeds) per arm on the baseline's own
# spectra, run windows, adapted library and mass calibration, so arms differ only in the
# candidate allowlist.
#   ps_val.sh ROOT BIN MZML LIBP LIBF CFG THREADS
set -u
ROOT=$1; BIN=$2; MZ=$3; LIBP=$4; LIBF=$5; CFG=$6; THREADS=$7
SEEDS=${SEEDS:-"0 1 2"}
ARMS=${ARMS:-"OFF PRESCAN CAP300 SEN BAL STR AGG"}
export MUMDIA_CACHE_DIR=off
mkdir -p "$ROOT"
log() { echo "$(date +%T) $*" >> "$ROOT/progress.log"; }
T() {  # T NAME CMD...: time a command into ROOT/time_NAME.txt and ROOT/NAME.log
  local name=$1; shift
  /usr/bin/time -v -o "$ROOT/time_$name.txt" "$@" > "$ROOT/$name.log" 2>&1
  local rc=$?; log "$name exit $rc"; return $rc
}
log "START bin=$(sha256sum "$BIN" | cut -c1-16) mz=$(basename "$MZ")"
B=$ROOT/BASE
if [ ! -f "$B/psms_scored.parquet" ]; then
  T base "$BIN" --threads "$THREADS" run --mzml "$MZ" --lib-precursors "$LIBP" \
    --lib-fragments "$LIBF" --out-dir "$B" --config "$CFG" || exit 1
fi
LIB=$LIBP
for f in multihead ft deeplc; do
  [ -f "$B/fragment_library_precursors_$f.parquet" ] && LIB=$B/fragment_library_precursors_$f.parquet
done
log "library for the chain: $LIB"
# Existing prescan arm: anchor every trimer (the tag screen's search-wide setting).
python3 - "$CFG" "$ROOT/cfg_prescan.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); c.setdefault("prescan", {})["anchor_all"] = True
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
for arm in $ARMS; do
  A=$ROOT/$arm; mkdir -p "$A"; R=""
  case $arm in
    OFF) ;;
    PRESCAN) T "${arm}_screen" "$BIN" --threads "$THREADS" prescan --ms2 "$B/spectra/spectra_ms2.parquet" \
        --isolation-windows "$B/spectra/isolation_windows.parquet" --lib-precursors "$LIB" \
        --run-windows "$B/run_windows.parquet" --out "$A/survivors.parquet" \
        --config "$ROOT/cfg_prescan.json" && R="--restrict-candidates $A/survivors.parquet" || continue ;;
    *) case $arm in CAP300) P="--preset balanced --top-peaks 300";; SEN) P="--preset sensitive";;
         BAL) P="--preset balanced";; STR) P="--preset stringent";; AGG) P="--preset aggressive";; esac
       T "${arm}_screen" "$BIN" --threads "$THREADS" prescreen --ms2 "$B/spectra/spectra_ms2.parquet" \
         --lib-precursors "$LIB" --run-windows "$B/run_windows.parquet" --out "$A/survivors.parquet" \
         --config "$CFG" --write-scores $P && R="--restrict-candidates $A/survivors.parquet" || continue ;;
  esac
  T "${arm}_extract" "$BIN" --threads "$THREADS" extract --ms2 "$B/spectra/spectra_ms2.parquet" \
    --ms1 "$B/spectra/spectra_ms1.parquet" --lib-precursors "$LIB" --lib-fragments "$LIBF" \
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
  for s in $SEEDS; do
    mkdir -p "$A/s$s"
    ( export MUMDIA_NN_SEED=$s; T "${arm}_rescore_s$s" "$BIN" --threads 16 rescore \
        --competed "$A/psms_competed.parquet" --out "$A/s$s/psms_scored.parquet" \
        --work-dir "$A/s$s/work" --config "$CFG"; rm -rf "$A/s$s/work" ) &
  done
done
wait
log "ALL_DONE"
touch "$ROOT/ALL_DONE"
