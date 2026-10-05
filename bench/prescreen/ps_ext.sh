#!/bin/bash
# Extension arms of the prescreen validation, after ps_val.sh finished in ROOT.
#   ps_ext.sh ROOT BIN LIBF CFG THREADS [OXMOD]
# X_* arms: full chain (extract -> features -> compete -> rescore x seeds) on the screen's survivors.
# S_* arms: evaluation sample (--sample-candidates), screen only; retention is measured against
#           the OFF arm's accepted precursors inside the sample.
set -u
ROOT=$1; BIN=$2; LIBF=$3; CFG=$4; THREADS=$5; OXMOD=${6:-M:Oxidation}
SEEDS=${SEEDS:-"0 1 2"}
SAMPLE=${SAMPLE:-100000}
export MUMDIA_CACHE_DIR=off
B=$ROOT/BASE; S=$B/spectra
LIB=$(grep "library for the chain" "$ROOT/progress.log" | tail -1 | sed 's/.*chain: //')
log() { echo "$(date +%T) $*" >> "$ROOT/progress.log"; }
T() {
  local name=$1; shift
  /usr/bin/time -v -o "$ROOT/time_$name.txt" "$@" > "$ROOT/$name.log" 2>&1
  local rc=$?; log "$name exit $rc"; return $rc
}
mkcfg() {  # mkcfg OUT JSON-OVERRIDES (merged into prescreen)
  python3 - "$CFG" "$1" "$2" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); o = json.loads(sys.argv[3])
def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict): merge(a.setdefault(k, {}), v)
        else: a[k] = v
merge(c.setdefault("prescreen", {}), o)
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
}
chain() {  # chain ARM
  local arm=$1 A=$ROOT/$1
  T "${arm}_extract" "$BIN" --threads "$THREADS" extract --ms2 "$S/spectra_ms2.parquet" \
    --ms1 "$S/spectra_ms1.parquet" --lib-precursors "$LIB" --lib-fragments "$LIBF" \
    --run-windows "$B/run_windows.parquet" --mass-cal "$B/seed_psms.parquet.masscal.json" \
    --out-psms "$A/psms_extracted.parquet" --out-chromatograms "$A/chromatograms.parquet" \
    --restrict-candidates "$A/survivors.parquet" --config "$CFG" || return
  T "${arm}_features" "$BIN" --threads "$THREADS" features --psms-extracted "$A/psms_extracted.parquet" \
    --chromatograms "$A/chromatograms.parquet" --seed-psms "$B/seed_psms.parquet" \
    --out "$A/features.parquet" --out-pin "$A/run.pin" --config "$CFG" || return
  rm -f "$A/chromatograms.parquet"
  T "${arm}_compete" "$BIN" --threads "$THREADS" compete --features "$A/features.parquet" \
    --out "$A/psms_competed.parquet" --config "$CFG" || return
  rm -f "$A/features.parquet"
  for s in $SEEDS; do
    mkdir -p "$A/s$s"
    ( export MUMDIA_NN_SEED=$s; T "${arm}_rescore_s$s" "$BIN" --threads 16 rescore \
        --competed "$A/psms_competed.parquet" --out "$A/s$s/psms_scored.parquet" \
        --work-dir "$A/s$s/work" --config "$CFG"; rm -rf "$A/s$s/work" ) &
  done
}
screen() {  # screen ARM OVERRIDES [extra args]
  local arm=$1 o=$2; shift 2
  mkdir -p "$ROOT/$arm"; mkcfg "$ROOT/$arm/cfg.json" "$o"
  T "${arm}_screen" "$BIN" --threads "$THREADS" prescreen --ms2 "$S/spectra_ms2.parquet" \
    --ms1 "$S/spectra_ms1.parquet" --lib-precursors "$LIB" --run-windows "$B/run_windows.parquet" \
    --out "$ROOT/$arm/survivors.parquet" --config "$ROOT/$arm/cfg.json" --write-scores "$@"
}
log "EXT START bin=$(sha256sum "$BIN" | cut -c1-16)"
# ---- full-chain arms ----
screen X_FAM '{"localization":"family_support"}' && chain X_FAM
screen X_BEST '{"localization":"best_site"}' && chain X_BEST
screen X_OX53 "{\"preset\":\"custom\",\"target\":0.53,\"scope\":\"modified\",\"scope_mods\":[\"$OXMOD\"]}" && chain X_OX53
screen X_RET '{"retrieval":"tags","rescue":false,"delayed_modforms":true}' && chain X_RET
screen X_RETR '{"retrieval":"tags","rescue":true}'
# ---- sampled-candidate arms (screen only) ----
screen S_BASE '{}' --sample-candidates "$SAMPLE"
screen S_COMP54 '{"complement_bonus":0.25}' --sample-candidates "$SAMPLE"
screen S_COMP75 '{"complement_bonus":0.25,"preset":"stringent"}' --sample-candidates "$SAMPLE"
screen S_BASE75 '{"preset":"stringent"}' --sample-candidates "$SAMPLE"
screen S_TAG '{"tag_bonus":0.25}' --sample-candidates "$SAMPLE"
screen S_FASTA '{"tag_bonus":0.25,"fasta_bonus":0.25}' --sample-candidates "$SAMPLE"
screen S_FLANK '{"flank_bonus":0.25}' --sample-candidates "$SAMPLE"
screen S_TRACE '{"trace":{"enabled":true,"bonus":0.25}}' --sample-candidates "$SAMPLE"
screen S_TRACEU '{"trace":{"enabled":true,"bonus":0.25,"pooling":"unmerged","score":"coherent_fragments"}}' --sample-candidates "$SAMPLE"
screen S_TRACEF '{"trace":{"enabled":true,"bonus":0.25,"score":"coherent_fragments"}}' --sample-candidates "$SAMPLE"
screen S_MASS '{"mass_hypotheses":{"enabled":true,"ms1":true,"sample_spectra":512,"bonus":0.25}}' --sample-candidates "$SAMPLE"
screen S_GAP '{"retrieval":"tags","rescue":false,"tags":{"gap_edges":true}}' --sample-candidates "$SAMPLE"
screen S_RET '{"retrieval":"tags","rescue":false}' --sample-candidates "$SAMPLE"
wait
log "EXT_DONE"
touch "$ROOT/EXT_DONE"
