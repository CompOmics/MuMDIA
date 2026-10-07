#!/bin/bash
# Clean wall/RSS timing after ps_val.sh and ps_ext.sh: one process at a time, nothing else of the
# campaign running. Times extract without and with each preset's survivors, and the screens.
#   ps_time.sh ROOT BIN LIBF CFG THREADS
set -u
ROOT=$1; BIN=$2; LIBF=$3; CFG=$4; THREADS=$5
export MUMDIA_CACHE_DIR=off
B=$ROOT/BASE; S=$B/spectra; O=$ROOT/TIMING
LIB=$(grep "library for the chain" "$ROOT/progress.log" | tail -1 | sed 's/.*chain: //')
mkdir -p "$O"
while pgrep -f "$BIN .*rescore" > /dev/null; do sleep 20; done
t() { local name=$1; shift; /usr/bin/time -v -o "$O/time_$name.txt" "$@" > "$O/$name.log" 2>&1; echo "$(date +%T) $name exit $?" >> "$O/progress.log"; }
for rep in 1 2; do
  for arm in OFF CAP300 SEN BAL STR AGG; do
    R=""; [ "$arm" != OFF ] && R="--restrict-candidates $ROOT/$arm/survivors.parquet"
    t "extract_${arm}_r$rep" "$BIN" --threads "$THREADS" extract --ms2 "$S/spectra_ms2.parquet" \
      --ms1 "$S/spectra_ms1.parquet" --lib-precursors "$LIB" --lib-fragments "$LIBF" \
      --run-windows "$B/run_windows.parquet" --mass-cal "$B/seed_psms.parquet.masscal.json" \
      --out-psms "$O/psms.parquet" --out-chromatograms "$O/chrom.parquet" $R --config "$CFG"
  done
  for p in "CAP300 --preset balanced --top-peaks 300" "SEN --preset sensitive" "BAL --preset balanced" "AGG --preset aggressive"; do
    set -- $p; arm=$1; shift
    t "screen_${arm}_r$rep" "$BIN" --threads "$THREADS" prescreen --ms2 "$S/spectra_ms2.parquet" \
      --lib-precursors "$LIB" --run-windows "$B/run_windows.parquet" --out "$O/surv.parquet" \
      --config "$CFG" "$@"
  done
  t "prescan_r$rep" "$BIN" --threads "$THREADS" prescan --ms2 "$S/spectra_ms2.parquet" \
    --isolation-windows "$S/isolation_windows.parquet" --lib-precursors "$LIB" \
    --run-windows "$B/run_windows.parquet" --out "$O/surv.parquet" --config "$ROOT/cfg_prescan.json"
done
rm -f "$O/psms.parquet" "$O/chrom.parquet" "$O"/surv.parquet*
python3 - "$O" <<'EOF'
import glob, os, sys, collections
O = sys.argv[1]; rows = collections.defaultdict(list)
for f in sorted(glob.glob(f"{O}/time_*.txt")):
    name = os.path.basename(f)[5:-4].rsplit("_r", 1)[0]; w = rss = None
    for line in open(f):
        if "Elapsed (wall" in line:
            v = line.rsplit(" ", 1)[1].strip().split(":"); w = 0.0
            for x in v: w = w * 60 + float(x)
        if "Maximum resident" in line: rss = int(line.rsplit(" ", 1)[1]) / 1048576
    rows[name].append((w, rss))
with open(f"{O}/timing.txt", "w") as fh:
    for k, v in rows.items():
        line = f"{k:16s} " + "  ".join(f"{w:7.1f}s {r:5.1f}GB" for w, r in v)
        fh.write(line + "\n"); print(line)
EOF
touch "$O/DONE"
