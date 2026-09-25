#!/usr/bin/env bash
# End-to-end smoke test on the generated fixture. Run by CI, and runnable locally.
#
#   ci/smoke.sh [work_dir]
#
# Needs: a built `mumdia` binary (found automatically, or set MUMDIA_BIN) and a
# Python with pyarrow. No sidecar (arm 5c stands a stub in for the DeepLC worker), no
# network, no data file in the repository: the
# fixture is generated from `test_data/fixture.fasta` and from the library the
# engine itself builds out of it, so the planted peaks cannot disagree with the
# engine's mass model.
#
# What it covers that the Rust suite does not: mzML parsing (`convert`), the
# `digest -> peptidoforms -> predict-frag` library build, the `run` orchestrator
# and its manifest, RT calibration on real anchors, and `quant` and `report`
# writing files. The Rust integration test builds its inputs in process and starts
# at `extract`, so none of that was exercised before.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
work="${1:-${TMPDIR:-/tmp}/mumdia_smoke}"
cfg="configs/examples/native.json"

# Locate the release binary.
#
# The CARGO-REPORTED target directory comes first, because a redirected
# `build.target-dir` -- which `.cargo/config.toml.example` documents, to keep build
# artifacts out of a synced folder -- means `rust/mumdia/target/` can still hold a
# months-old binary from before the redirect. Not hypothetical: that is exactly how a
# writer/reader mismatch (convert moved to LargeList, spectra.rs still downcast to
# List) passed here and failed in CI. An in-tree path that shadows the real build
# output is worse than no path at all.
#
# `./mumdia` comes LAST, for the release archive, where the binary sits beside
# `ci/` and there is no cargo and no `rust/` tree. In a source checkout it would be
# a stray copy of unknown age, which is why it does not come first.
find_bin() {
    if [ -n "${MUMDIA_BIN:-}" ]; then echo "$MUMDIA_BIN"; return; fi
    target=""
    if command -v cargo > /dev/null 2>&1; then
        # Run cargo from INSIDE the workspace, not with --manifest-path from the repo
        # root: cargo discovers `.cargo/config.toml` by walking up from the CURRENT
        # DIRECTORY, not from the manifest, so from the root it reports the in-tree
        # `target/` and misses the redirect entirely. That is the bug this whole
        # function exists to avoid, so getting it wrong here defeats the point.
        target=$( (cd rust/mumdia && cargo metadata --format-version 1 --no-deps 2>/dev/null) \
                 | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])' \
                 2>/dev/null || true)
    fi
    for cand in \
        "${target:+$target/release/mumdia}" \
        "${target:+$target/release/mumdia.exe}" \
        rust/mumdia/target/release/mumdia \
        rust/mumdia/target/release/mumdia.exe \
        ./mumdia \
        ./mumdia.exe
    do
        [ -n "$cand" ] && [ -x "$cand" ] && { echo "$cand"; return; }
    done
    command -v mumdia || { echo "no mumdia binary found; set MUMDIA_BIN" >&2; exit 2; }
}

# Before find_bin, which shells out to it.
PY="${PYTHON:-python}"
BIN="$(find_bin)"
echo "=== smoke: binary $BIN"
"$BIN" --version
rm -rf "$work"
mkdir -p "$work"

# The engine must be able to say the configuration is runnable before we run it.
echo "=== smoke: doctor"
"$BIN" doctor --config "$cfg"

# 1. Library, standalone, so the fixture generator can read the exact masses the
#    engine will look for.
echo "=== smoke: build the fixture library"
"$BIN" digest --fasta test_data/fixture.fasta --out "$work/peptides.parquet" --config "$cfg"
"$BIN" peptidoforms --peptides "$work/peptides.parquet" --out "$work/peptidoforms.parquet" \
    --config "$cfg"
"$BIN" predict-frag --peptidoforms "$work/peptidoforms.parquet" \
    --out-precursors "$work/lib_prec.parquet" --out-fragments "$work/lib_frag.parquet" \
    --work-dir "$work/sidecar" --config "$cfg"

# 2. The mzML.
echo "=== smoke: generate the fixture mzML"
# --quiet rather than `| head -3`: truncating the generator's output closes its
# stdout early, and on a Windows console that surfaces as
# `OSError: [Errno 22] Invalid argument` during interpreter shutdown rather than as
# EPIPE, so the script exited 120 with no failing assertion to point at. It passed
# on Linux and locally and failed only on the CI Windows runner.
"$PY" ci/make_fixture_mzml.py \
    --precursors "$work/lib_prec.parquet" --fragments "$work/lib_frag.parquet" \
    --out "$work/fixture.mzML" --manifest "$work/planted.json" \
    --n-planted 160 --windows 8 --quiet

# 3. The full single-run orchestrator, from the FASTA, so digest runs inside `run`
#    as well and the library it builds is compared against the one the fixture was
#    generated from by construction.
echo "=== smoke: run"
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out" --config "$cfg" --threads 2

# 4. A second run into a fresh directory, to assert byte-level determinism.
echo "=== smoke: run again for determinism"
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out2" --config "$cfg" --threads 2 > "$work/run2.log" 2>&1 \
    || { tail -20 "$work/run2.log"; exit 1; }

# 4b. A malformed retention time must not abort the run.
#
#     Regression test for a reproduced crash: one `NaN` scan start time in one scan of
#     an mzML passed `convert` unchecked, was written into the spectra artifact, and
#     aborted `mumdia run` inside extract with `called `Option::unwrap()` on a `None`
#     value`, naming neither the file, nor the scan, nor the value. The peak filter had
#     this hole closed for m/z and intensity; retention time was missed, and it is the
#     value every later stage keys on.
#
#     Asserts the three things that matter: the run completes, convert says what it
#     dropped, and the result is the same as from the clean file (the poisoned scan is
#     one of 480, and dropping it must not change which peptides are found).
echo "=== smoke: a NaN retention time must be dropped, not fatal"
"$PY" - "$work/fixture.mzML" "$work/fixture_nan_rt.mzML" <<'PYEOF'
import io, sys
src, dst = sys.argv[1], sys.argv[2]
s = io.open(src, encoding="utf-8").read()
marker = 'scan start time" value="0.003704"'
if marker not in s:
    sys.exit(f"fixture no longer contains {marker!r}; update this check")
io.open(dst, "w", encoding="utf-8", newline="").write(
    s.replace(marker, 'scan start time" value="NaN"', 1))
PYEOF
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture_nan_rt.mzML" \
    --out-dir "$work/out_nan_rt" --config "$cfg" --threads 2 \
    > "$work/run_nan_rt.log" 2>&1 \
    || { echo "a single NaN retention time aborted the run"; tail -20 "$work/run_nan_rt.log"; exit 1; }
grep -q "nonfinite_rt=1" "$work/run_nan_rt.log" \
    || { echo "convert did not report the dropped spectrum"; exit 1; }
diff <(cut -f1-3 "$work/out/peptides.tsv") <(cut -f1-3 "$work/out_nan_rt/peptides.tsv") \
    > /dev/null \
    || { echo "dropping one unusable scan changed the identifications"; exit 1; }
echo "    ok: run completed, drop reported, identifications unchanged"

# 4c. The GROUPED orchestrator (`groups.window_groups > 1`), which nothing covered.
#
#     `run_groups` is a second orchestrator: it cuts the library into m/z bands, seeds,
#     calibrates, extracts, features and competes each band separately, and pools the
#     results back into the artifacts the ungrouped path writes. Every test in the Rust
#     suite and every arm above runs the UNGROUPED path, so a wiring regression in
#     run_groups -- a band lent the wrong buffer, a stale buffer, a band pooled out of
#     order -- passed the whole suite. It is checked here rather than as a Rust
#     integration test because the tiny hand-crafted fixture in `tests/pipeline.rs` cannot
#     drive it: run_groups needs converted spectra with several isolation windows, a
#     library with row-group statistics on `precursor_mz`, and a retention-time model
#     fitted on confident seed anchors, all of which this fixture already has and that one
#     would have to fake.
#
#     `calibration: per_group` on purpose. It is the mode in which each band applies its
#     OWN mass calibration to the run's spectra, which since the bands share one decoded
#     buffer is the configuration most exposed to a stage writing a corrected m/z back
#     into the scans.
echo "=== smoke: the grouped orchestrator"
cat > "$work/grouped.json" <<'JSONEOF'
{
  "features": { "set": "extended" },
  "extract": { "apex_count_window": 5, "apex_rt_prior_s": 120.0, "gate_min_score": 0.2 },
  "groups": { "window_groups": 3, "parallel": 2, "calibration": "per_group" }
}
JSONEOF
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out_grouped" --config "$work/grouped.json" --threads 4 \
    > "$work/grouped.log" 2>&1 \
    || { tail -30 "$work/grouped.log"; echo "the grouped run failed"; exit 1; }
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out_grouped2" --config "$work/grouped.json" --threads 4 \
    > "$work/grouped2.log" 2>&1 \
    || { tail -30 "$work/grouped2.log"; echo "the second grouped run failed"; exit 1; }

# Three bands were planned, and each writes its own competed table.
grep -q '"window_groups": 3' "$work/out_grouped/groups/plan.json" \
    || { echo "the grouped run did not plan three bands"; exit 1; }
for g in g00 g01 g02; do
    test -s "$work/out_grouped/groups/$g/psms_competed.parquet" \
        || { echo "band $g produced no competed table"; exit 1; }
done

# The run's spectra are decoded ONCE PER PHASE, not once per band. Before the buffers were
# shared each band's search-seed and each band's extract opened the artifact itself, which
# on three bands was six MS2 decodes and three MS1 decodes; the counts below are what
# regresses if a band is ever handed its own copy again.
# No `$` anchor: the engine's log lines end CRLF on a Windows runner, so an
# end-of-line anchor after the stage name matches nothing there. Both names are
# unique substrings in the log as it is.
n_ms2=$(grep -c 'stage=load-ms2' "$work/grouped.log" || true)
n_ms1=$(grep -c 'stage=load-ms1' "$work/grouped.log" || true)
[ "$n_ms2" = "2" ] || { echo "expected 2 MS2 decodes for 3 bands, saw $n_ms2"; exit 1; }
[ "$n_ms1" = "1" ] || { echo "expected 1 MS1 decode for 3 bands, saw $n_ms1"; exit 1; }

# The grouped path identifies peptides, and does so reproducibly. Not compared against the
# ungrouped run: banding changes the competition population and the pooled q denominator,
# so the two are not expected to agree row for row (measured on this fixture: 150 rows
# ungrouped, 146 grouped under per_group).
n_grouped=$(($(wc -l < "$work/out_grouped/peptides.tsv") - 1))
[ "$n_grouped" -ge 100 ] \
    || { echo "the grouped run found only $n_grouped peptides; expected at least 100"; exit 1; }
diff "$work/out_grouped/peptides.tsv" "$work/out_grouped2/peptides.tsv" > /dev/null \
    || { echo "two grouped runs of the same input disagree"; exit 1; }
echo "    ok: 3 bands, $n_ms2 MS2 + $n_ms1 MS1 decodes, $n_grouped peptides, reproducible"

# The same grouped run with the competed rows and the chromatograms left per band
# (`groups.pool_competed = false`, `groups.pool_chromatograms = false`): rescore reads the
# three band tables with a table-to-source map instead of the pooled copy, which must give
# the same scored table byte for byte, and quant reads the three band chromatogram tables
# with the overlap losers, which must give the same quant tables byte for byte. The
# fixture's bands do not overlap, so the competed option applies; the log lines and the
# absent pooled tables prove both did.
echo "=== smoke: the grouped run with the competed rows and chromatograms left per band"
sed 's/"calibration": "per_group" }/"calibration": "per_group", "pool_competed": false, "pool_chromatograms": false }/' \
    "$work/grouped.json" > "$work/grouped_nopool.json"
grep -q '"pool_competed": false' "$work/grouped_nopool.json" \
    || { echo "could not derive the pool_competed = false config"; exit 1; }
grep -q '"pool_chromatograms": false' "$work/grouped_nopool.json" \
    || { echo "could not derive the pool_chromatograms = false config"; exit 1; }
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out_grouped_nopool" --config "$work/grouped_nopool.json" --threads 4 \
    > "$work/grouped_nopool.log" 2>&1 \
    || { tail -30 "$work/grouped_nopool.log"; echo "the grouped run without a pooled competed table failed"; exit 1; }
grep -q 'the competed rows stay per band' "$work/grouped_nopool.log" \
    || { echo "groups.pool_competed = false did not leave the competed rows per band"; exit 1; }
test ! -e "$work/out_grouped_nopool/psms_competed.parquet" \
    || { echo "a pooled psms_competed.parquet was written under pool_competed = false"; exit 1; }
cmp -s "$work/out_grouped/psms_scored.parquet" "$work/out_grouped_nopool/psms_scored.parquet" \
    || { echo "rescoring the band tables changed psms_scored.parquet"; exit 1; }
grep -q 'the chromatograms stay per band' "$work/grouped_nopool.log" \
    || { echo "groups.pool_chromatograms = false did not leave the chromatograms per band"; exit 1; }
test ! -e "$work/out_grouped_nopool/chromatograms.parquet" \
    || { echo "a pooled chromatograms.parquet was written under pool_chromatograms = false"; exit 1; }
test -s "$work/out_grouped_nopool/groups/overlap_losers.parquet" \
    || { echo "the overlap losers were not persisted under pool_chromatograms = false"; exit 1; }
for f in peptide_quant.parquet protein_group_quant.parquet fragment_quant.parquet peptides.tsv proteins.tsv; do
    cmp -s "$work/out_grouped/$f" "$work/out_grouped_nopool/$f" \
        || { echo "quantifying the band chromatogram tables changed $f"; exit 1; }
done
echo "    ok: band tables rescored and quantified; psms_scored and the quant tables byte-identical to the pooled run's"

# 4d. Two bands under the default `calibration: global`: the pooled fragment mass
#     calibration is the one the ungrouped run fitted (docs/33 section 4a). Until
#     2026-09-25 each band offered only its 2,000 best targets as calibrants, which was
#     short at 2 and 4 bands on real data; every target is offered now. The fixture's
#     bands hold fewer than 2,000 targets, so this pins the equality on real spectra and
#     the test `a_two_band_pooled_mass_calibration_equals_the_unbanded_fit`
#     (`tests/pipeline.rs`) pins the case the prefix got wrong.
echo "=== smoke: two bands, global calibration, pooled mass calibration"
cat > "$work/grouped_2b.json" <<'JSONEOF'
{
  "features": { "set": "extended" },
  "extract": { "apex_count_window": 5, "apex_rt_prior_s": 120.0, "gate_min_score": 0.2 },
  "groups": { "window_groups": 2, "calibration": "global" }
}
JSONEOF
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out_grouped_2b" --config "$work/grouped_2b.json" --threads 2 \
    > "$work/grouped_2b.log" 2>&1 \
    || { tail -30 "$work/grouped_2b.log"; echo "the two-band run failed"; exit 1; }
"$PY" - "$work/out/seed_psms.parquet.masscal.json" \
    "$work/out_grouped_2b/seed_psms.parquet.masscal.json" <<'PYEOF'
import json, sys
one = json.load(open(sys.argv[1], encoding="utf-8"))
two = json.load(open(sys.argv[2], encoding="utf-8"))
if two.get("masscal_source") != "pooled_deviations":
    sys.exit("the two-band run did not fit its mass calibration on the pooled deviations: %r"
             % two.get("masscal_source"))
if one["n_dev"] != two["n_dev"]:
    sys.exit("the pooled mass calibration saw %d calibrant deviations, the ungrouped run %d"
             % (two["n_dev"], one["n_dev"]))
for key in ("frag_ppm_offset", "frag_tol_ppm"):
    a, b = float(one[key]), float(two[key])
    # The sidecar stores deviations as f32, so the two agree to f32 precision.
    if abs(a - b) > 1e-5 * max(1.0, abs(a)):
        sys.exit("%s: pooled %r against ungrouped %r" % (key, b, a))
print("    ok: 2 bands, %d calibrant deviations, frag_tol_ppm %.6g, as ungrouped"
      % (two["n_dev"], float(two["frag_tol_ppm"])))
PYEOF

# 4e. The chromatogram v2 layout (`extract.chromatogram_schema = 2`, docs/15 "Layout v2"):
#     the RT axis once per candidate per row group, each trace trimmed to its nonzero run.
#     Every reader rebuilds the v1 rows from it, so every table after extract must be the
#     v1 run's in `$work/out`, byte for byte, while the chromatogram table says schema 2
#     and stores its lists as `rt_axis`/`intensity_trimmed`, never as `rt`/`intensity`.
#     A second v2 run puts a row-group seam at EVERY row (`MUMDIA_CHROM_ROW_GROUP_ROWS=1`,
#     a test knob that moves only the seams), so no row may take its axis from another
#     row and every read that starts at a seam must still find one; the same bytes again.
#     Then the grouped run of 4c under v2, pooled and per band: the pool splices v2 band
#     tables (re-encoding the groups that hold an overlap loser) and quant reads either.
echo "=== smoke: chromatograms v2 leave every downstream table byte-identical"
"$PY" - "$cfg" "$work/chrom_v2.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
c.setdefault("extract", {})["chromatogram_schema"] = 2
json.dump(c, open(sys.argv[2], "w", encoding="utf-8"), indent=2)
PYEOF
"$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$work/out_chrom_v2" --config "$work/chrom_v2.json" --threads 2 \
    > "$work/chrom_v2.log" 2>&1 \
    || { tail -30 "$work/chrom_v2.log"; echo "the chromatograms v2 run failed"; exit 1; }
MUMDIA_CHROM_ROW_GROUP_ROWS=1 "$BIN" run --fasta test_data/fixture.fasta \
    --mzml "$work/fixture.mzML" --out-dir "$work/out_chrom_v2_rg1" \
    --config "$work/chrom_v2.json" --threads 2 > "$work/chrom_v2_rg1.log" 2>&1 \
    || { tail -30 "$work/chrom_v2_rg1.log"; echo "the v2 run with a seam at every row failed"; exit 1; }
grep -q 'chromatogram row groups resized' "$work/chrom_v2_rg1.log" \
    || { echo "MUMDIA_CHROM_ROW_GROUP_ROWS did not reach extract"; exit 1; }
for d in out_chrom_v2 out_chrom_v2_rg1; do
    for f in psms_extracted.parquet features.parquet psms_competed.parquet psms_scored.parquet \
             peptide_quant.parquet protein_group_quant.parquet fragment_quant.parquet \
             peptides.tsv proteins.tsv; do
        cmp -s "$work/out/$f" "$work/$d/$f" \
            || { echo "chromatograms v2 ($d) changed $f"; exit 1; }
    done
done
"$PY" - "$work/grouped.json" "$work/grouped_v2.json" "$work/grouped_nopool_v2.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
c.setdefault("extract", {})["chromatogram_schema"] = 2
json.dump(c, open(sys.argv[2], "w", encoding="utf-8"), indent=2)
c["groups"]["pool_chromatograms"] = False
json.dump(c, open(sys.argv[3], "w", encoding="utf-8"), indent=2)
PYEOF
for arm in grouped_v2 grouped_nopool_v2; do
    "$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
        --out-dir "$work/out_$arm" --config "$work/$arm.json" --threads 4 \
        > "$work/$arm.log" 2>&1 \
        || { tail -30 "$work/$arm.log"; echo "the grouped chromatograms v2 run ($arm) failed"; exit 1; }
    for f in psms_scored.parquet peptide_quant.parquet protein_group_quant.parquet \
             fragment_quant.parquet peptides.tsv proteins.tsv; do
        cmp -s "$work/out_grouped/$f" "$work/out_$arm/$f" \
            || { echo "chromatograms v2 changed $f of the grouped run ($arm)"; exit 1; }
    done
done
"$PY" - "$work" <<'PYEOF'
import json, os, sys
import pyarrow.parquet as pq
work = sys.argv[1]
def rec(d):
    m = json.load(open(os.path.join(work, d, "manifest.json"), encoding="utf-8"))
    return m["artifacts"]["chromatograms"]
def check(ok, what):
    if not ok:
        sys.exit("chromatograms v2: " + what)
v1 = os.path.join(work, "out", "chromatograms.parquet")
v2 = os.path.join(work, "out_chrom_v2", "chromatograms.parquet")
rg1 = os.path.join(work, "out_chrom_v2_rg1", "chromatograms.parquet")
check(rec("out")["schema_version"] == 1, "the default run no longer records schema 1")
for d in ("out_chrom_v2", "out_chrom_v2_rg1", "out_grouped_v2"):
    check(rec(d)["schema_version"] == 2, f"{d} does not record chromatograms schema 2")
f1, f2, fr = pq.ParquetFile(v1), pq.ParquetFile(v2), pq.ParquetFile(rg1)
names = lambda f: f.schema_arrow.names
V1_LISTS = {"rt", "intensity"}
V2_COLS = {"rt_axis", "intensity_trimmed", "trace_offset", "trace_len"}
check(V1_LISTS <= set(names(f1)) and not V2_COLS & set(names(f1)),
      "the default table is not v1: it lacks rt/intensity or has a v2 column")
# The v2 lists are renamed so that a reader that knows only v1 stops at the missing `rt`
# instead of taking an empty axis beside a trimmed trace for an observed row.
for f, d in ((f2, "out_chrom_v2"), (fr, "out_chrom_v2_rg1")):
    check(V2_COLS <= set(names(f)) and not V1_LISTS & set(names(f)),
          f"the {d} table does not have the v2 columns in place of rt/intensity")
check(f1.metadata.num_rows == f2.metadata.num_rows == fr.metadata.num_rows,
      "the layouts hold different row counts")
check(all(fr.metadata.row_group(i).num_rows == 1 for i in range(fr.metadata.num_row_groups)),
      "MUMDIA_CHROM_ROW_GROUP_ROWS=1 did not put a seam at every row")
t1 = pq.read_table(v1, columns=["rt", "intensity"])
t2 = pq.read_table(v2, columns=["rt_axis", "intensity_trimmed"])
vals = lambda t, c: sum(len(x) for x in t.column(c).to_pylist())
s1, s2 = os.path.getsize(v1), os.path.getsize(v2)
print("    ok: %d rows; rt values %d -> %d, intensity values %d -> %d; %d -> %d bytes (%.1f%% smaller)"
      % (f1.metadata.num_rows, vals(t1, "rt"), vals(t2, "rt_axis"), vals(t1, "intensity"),
         vals(t2, "intensity_trimmed"), s1, s2, 100.0 * (1 - s2 / s1)))
PYEOF
echo "    ok: every table after extract byte-identical to v1, ungrouped (default seams and a seam at every row) and grouped (pooled and per band)"

# 5. The multi-run orchestrator. Nothing tested it: `run-experiment` has a pooled
#    rescore, a by-source split, per-run quant and a cross-run LFQ that the
#    single-run path never reaches, and a split that drops rows produces plausible
#    numbers rather than an error. Two copies of the same fixture also make the
#    per-run outputs directly comparable: identical input must give identical rows.
echo "=== smoke: run-experiment over two runs"
cp "$work/fixture.mzML" "$work/fixture_b.mzML"
"$BIN" run-experiment --fasta test_data/fixture.fasta \
    --mzml "$work/fixture.mzML" --mzml "$work/fixture_b.mzML" \
    --run-names a --run-names b \
    --out-dir "$work/exp" --config "$cfg" > "$work/exp.log" 2>&1 \
    || { tail -30 "$work/exp.log"; exit 1; }
# The experiment-wide report once more, at a looser threshold, through the standalone
# command: on this fixture the pooled peptide-level q never reaches 1 percent (one decoy
# peptide against 151 targets is 1.3 percent), so the root pair run-experiment writes is
# header-only at the default threshold and the rows are asserted on this copy instead.
"$BIN" report --experiment-dir "$work/exp" --out-dir "$work/exp_report" --q 0.05 \
    > "$work/exp_report.log" 2>&1 || { tail -20 "$work/exp_report.log"; exit 1; }

# 5b. The same experiment with `experiment.parallel_runs = 2`, and one whose second run
#     cannot be converted.
#
#     An ungrouped run-experiment runs in three phases: convert every run, seed every run
#     against ONE shared seed library, then continue each run's chain. With
#     parallel_runs > 1 the converts, the seeds (all reading the one lent library) and the
#     chains run concurrently, which nothing else exercises. Every stage is deterministic
#     and reads only its own run's inputs, so every parquet and TSV must be the sequential
#     experiment's, byte for byte (the report JSONs and manifests carry wall clocks and
#     are left out). Then the documented consequence of the phase order: a conversion
#     failure of run 2 stops the experiment before run 1 is seeded or extracted, under
#     both settings.
echo "=== smoke: run-experiment with parallel_runs = 2, and a failing conversion"
"$PY" - "$cfg" "$work/exp_par.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
c.setdefault("experiment", {})["parallel_runs"] = 2
json.dump(c, open(sys.argv[2], "w"), indent=2)
PYEOF
"$BIN" run-experiment --fasta test_data/fixture.fasta \
    --mzml "$work/fixture.mzML" --mzml "$work/fixture_b.mzML" \
    --run-names a --run-names b \
    --out-dir "$work/exp_par" --config "$work/exp_par.json" > "$work/exp_par.log" 2>&1 \
    || { tail -30 "$work/exp_par.log"; echo "run-experiment with parallel_runs = 2 failed"; exit 1; }
grep -q "per-run chains in parallel" "$work/exp_par.log" \
    || { echo "parallel_runs = 2 did not take the parallel path"; exit 1; }
(cd "$work/exp" && find . -type f \( -name '*.parquet' -o -name '*.tsv' \) | sort) \
    > "$work/exp_files.txt"
(cd "$work/exp_par" && find . -type f \( -name '*.parquet' -o -name '*.tsv' \) | sort) \
    > "$work/exp_par_files.txt"
diff "$work/exp_files.txt" "$work/exp_par_files.txt" > /dev/null \
    || { echo "parallel_runs = 2 wrote a different set of tables"; \
         diff "$work/exp_files.txt" "$work/exp_par_files.txt"; exit 1; }
n_tables=0
while read -r f; do
    cmp -s "$work/exp/$f" "$work/exp_par/$f" \
        || { echo "parallel_runs = 2 changed $f"; exit 1; }
    n_tables=$((n_tables + 1))
done < "$work/exp_files.txt"
[ "$n_tables" -ge 20 ] \
    || { echo "only $n_tables experiment tables compared; expected at least 20"; exit 1; }
# `parallel_runs = "auto"`: the other scheduler (one run per 16 threads, each chain in a
# thread pool of its own, pulled from a queue). At --threads 32 it runs both chains at once
# in two 16-thread pools, and on Linux the first chain runs alone to be measured first.
# Every table must still be the sequential experiment's.
"$PY" - "$cfg" "$work/exp_auto.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
c.setdefault("experiment", {})["parallel_runs"] = "auto"
json.dump(c, open(sys.argv[2], "w"), indent=2)
PYEOF
"$BIN" run-experiment --fasta test_data/fixture.fasta \
    --mzml "$work/fixture.mzML" --mzml "$work/fixture_b.mzML" \
    --run-names a --run-names b --threads 32 \
    --out-dir "$work/exp_auto" --config "$work/exp_auto.json" > "$work/exp_auto.log" 2>&1 \
    || { tail -30 "$work/exp_auto.log"; echo "run-experiment with parallel_runs = auto failed"; exit 1; }
grep -q "each in a pool of its own (parallel_runs = auto)" "$work/exp_auto.log" \
    || { echo "parallel_runs = auto did not take the pooled path"; exit 1; }
(cd "$work/exp_auto" && find . -type f \( -name '*.parquet' -o -name '*.tsv' \) | sort) \
    > "$work/exp_auto_files.txt"
diff "$work/exp_files.txt" "$work/exp_auto_files.txt" > /dev/null \
    || { echo "parallel_runs = auto wrote a different set of tables"; \
         diff "$work/exp_files.txt" "$work/exp_auto_files.txt"; exit 1; }
while read -r f; do
    cmp -s "$work/exp/$f" "$work/exp_auto/$f" \
        || { echo "parallel_runs = auto changed $f"; exit 1; }
done < "$work/exp_files.txt"
printf 'this is not an mzML file\n' > "$work/fixture_bad.mzML"
for exp_cfg in "$cfg" "$work/exp_par.json" "$work/exp_auto.json"; do
    rm -rf "$work/exp_fail"
    if "$BIN" run-experiment --fasta test_data/fixture.fasta \
        --mzml "$work/fixture.mzML" --mzml "$work/fixture_bad.mzML" \
        --run-names a --run-names b \
        --out-dir "$work/exp_fail" --config "$exp_cfg" > "$work/exp_fail.log" 2>&1; then
        echo "run-experiment accepted an mzML it cannot convert ($exp_cfg)"; exit 1
    fi
    grep -q "fixture_bad.mzML" "$work/exp_fail.log" \
        || { tail -20 "$work/exp_fail.log"; echo "the failure does not name the bad file"; exit 1; }
    test -s "$work/exp_fail/a/spectra/spectra_ms2.parquet" \
        || { echo "run a was not converted before run b failed ($exp_cfg)"; exit 1; }
    for f in seed_psms.parquet psms_extracted.parquet; do
        [ ! -e "$work/exp_fail/a/$f" ] \
            || { echo "run a reached $f although run b's conversion failed ($exp_cfg)"; exit 1; }
    done
done
echo "    ok: $n_tables tables byte-identical under parallel_runs = 2 and auto; a failed conversion stops before any seed"

# 5c. The DeepLC branches of the orchestrators, with a stub worker.
#
#     CI has no DeepLC, so every choice `run_groups` and `run-experiment` make about WHICH
#     DeepLC call to make, and where its output goes, ran only on a developer machine:
#     `groups.rt_adaptation = once_per_run` writing each band under the name a later run's
#     `shared_bands` looks for, the bands keeping an experiment-level re-prediction without
#     calling the worker, a later run taking the first grouped run's band slices, the refusal
#     of a deferred library the calibration did not fully re-predict, and
#     `experiment.overlap_front_threads`. A regression there (a later run failing with
#     "shared bands do not match", or a calibration silently skipped) passed CI.
#
#     The stub stands in for `deeplc_finetune.py` under the same positional contract: it
#     copies each `lib_in` to its `lib_out` (so every retention time is the library's own),
#     writes the `<lib_out>.summary.json` the engine reads, and appends one JSON line per call
#     to `MUMDIA_STUB_DEEPLC_LOG`. A `deeplc-4.5.0.dist-info` on `PYTHONPATH` answers the
#     engine's version check. What is asserted is the calls and the files, not retention
#     times.
echo "=== smoke: DeepLC branch selection with a stub worker"
stub="$work/stub"
rm -rf "$stub"
mkdir -p "$stub"
# Written by Python, so the interpreter path and the directories land in the JSON in the
# form the engine and the interpreter both read on every platform.
stub_env=$("$PY" - "$stub" <<'PYEOF'
import json, os, sys
root = os.path.abspath(sys.argv[1])
site = os.path.join(root, "site", "deeplc-4.5.0.dist-info")
scripts = os.path.join(root, "scripts")
os.makedirs(site)
os.makedirs(scripts)
with open(os.path.join(site, "METADATA"), "w", encoding="utf-8") as fh:
    fh.write("Metadata-Version: 2.1\nName: deeplc\nVersion: 4.5.0\n")
STUB = r'''"""Stand-in for deeplc_finetune.py in ci/smoke.sh (arm 5c). No DeepLC, no torch."""
import json, os, shutil, sys
import pyarrow.parquet as pq

VALUED = {"--multihead", "--q-train", "--window-holdout-frac", "--threads",
          "--predict-threads", "--shards", "--projection-cache", "--bands", "--epochs",
          "--patience", "--batch", "--seed"}
pos, opts, argv, i = [], {}, sys.argv[1:], 0
while i < len(argv):
    a = argv[i]
    if a in VALUED:
        opts[a] = argv[i + 1]
        i += 2
    elif a.startswith("--"):
        opts[a] = True
        i += 1
    else:
        pos.append(a)
        i += 1
lib_in, seed, lib_out = pos[:3]
if "--bands" in opts:
    with open(opts["--bands"], encoding="utf-8") as fh:
        pairs = [line.split("\t") for line in fh.read().splitlines() if line]
else:
    pairs = [[lib_in, lib_out]]
mode = ("multihead" if "--multihead" in opts else
        "repredict" if "--no-finetune" in opts else "finetune")
with open(os.environ["MUMDIA_STUB_DEEPLC_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"mode": mode, "bands": "--bands" in opts, "seed": seed,
                         "pairs": pairs}) + "\n")
if os.environ.get("MUMDIA_STUB_DEEPLC_FAIL"):
    sys.exit("stub DeepLC worker: failing as asked (MUMDIA_STUB_DEEPLC_FAIL)")
retain = int(os.environ.get("MUMDIA_STUB_DEEPLC_RETAIN", "0"))
for src, dst in pairs:
    shutil.copyfile(src, dst)
    rows = pq.read_metadata(src).num_rows
    with open(dst + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "repredicted": rows - retain, "retained_imported": retain,
                   "retained_non_standard": retain, "retained_no_prediction": 0,
                   "model": "stub"}, fh)
'''
with open(os.path.join(scripts, "deeplc_finetune.py"), "w", encoding="utf-8") as fh:
    fh.write(STUB)
# Present so the script directory resolves as one holding workers; never called here.
with open(os.path.join(scripts, "deeplc_worker.py"), "w", encoding="utf-8") as fh:
    fh.write('import sys\nsys.exit("stub: deeplc_worker.py is not stubbed")\n')

base = {
    "features": {"set": "extended"},
    "extract": {"apex_count_window": 5, "apex_rt_prior_s": 120.0, "gate_min_score": 0.2},
    "predict_frag": {"deeplc_python": sys.executable, "sidecar_script_dir": scripts},
}
def arm(name, **sections):
    cfg = json.loads(json.dumps(base))
    for key, value in sections.items():
        cfg.setdefault(key, {}).update(value)
    with open(os.path.join(root, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
banded = {"window_groups": 2, "calibration": "global"}
arm("union", groups=dict(banded, rt_adaptation="once_per_run"),
    rt_im_train={"multihead_calibration": 2})
arm("keep", groups=dict(banded, rt_adaptation="once_per_run"),
    rt_im_train={"multihead_calibration": 0})
arm("perband", groups=banded, rt_im_train={"multihead_calibration": 0})
arm("defer", groups=dict(banded, rt_adaptation="once_per_run"),
    rt_im_train={"multihead_calibration": 2},
    predict_frag={"rt_predictor": "deeplc", "defer_deeplc_to_multihead": True})
arm("seq", rt_im_train={"multihead_calibration": 2})
arm("overlap", rt_im_train={"multihead_calibration": 2},
    experiment={"overlap_front_threads": 1})
# No trailing newline: on Windows it would arrive as CR LF, and $( ) strips the LF only.
sys.stdout.write(os.path.join(root, "site"))
PYEOF
)
stub_site="${stub_env%$'\r'}"
# One stub call log per arm; each command below runs with the stub on PYTHONPATH only.
stub_run() {
    local name="$1"; shift
    PYTHONPATH="$stub_site" MUMDIA_STUB_DEEPLC_LOG="$stub/$name.calls.jsonl" "$@" \
        > "$stub/$name.log" 2>&1
}
stub_experiment() {
    local name="$1" cfgname="$2"; shift 2
    stub_run "$name" "$BIN" run-experiment \
        --lib-precursors "$work/lib_prec.parquet" --lib-fragments "$work/lib_frag.parquet" \
        --mzml "$work/fixture.mzML" --mzml "$work/fixture_b.mzML" \
        --run-names a --run-names b --out-dir "$stub/$name" \
        --config "$stub/$cfgname.json" --threads 2 "$@"
}
for a in union keep perband; do
    stub_experiment "$a" "$a" \
        || { tail -30 "$stub/$a.log"; echo "stub arm $a failed"; exit 1; }
done
stub_experiment seq seq || { tail -30 "$stub/seq.log"; echo "stub arm seq failed"; exit 1; }
stub_experiment overlap overlap \
    || { tail -30 "$stub/overlap.log"; echo "stub arm overlap failed"; exit 1; }
# A failing adaptation under the overlap: the run fails, and with the worker's error.
if MUMDIA_STUB_DEEPLC_FAIL=1 stub_experiment overlap_fail overlap; then
    echo "the overlap arm succeeded although its DeepLC worker failed"; exit 1
fi
grep -q "DeepLC multi-head calibration failed" "$stub/overlap_fail.log" \
    || { tail -20 "$stub/overlap_fail.log"; echo "the overlap arm did not report the worker's failure"; exit 1; }
# The deferred FASTA build: refused when the calibration keeps a row's placeholder iRT,
# accepted when it re-predicts every row.
stub_run defer_fast "$BIN" run --fasta test_data/fixture.fasta --mzml "$work/fixture.mzML" \
    --out-dir "$stub/defer_fast" --config "$stub/defer.json" --threads 2 \
    || { tail -30 "$stub/defer_fast.log"; echo "the deferred-DeepLC grouped run failed"; exit 1; }
grep -q "the deferred DeepLC pass was not needed" "$stub/defer_fast.log" \
    || { echo "the deferred-DeepLC run did not check its bands"; exit 1; }
if MUMDIA_STUB_DEEPLC_RETAIN=1 stub_run defer_kept "$BIN" run --fasta test_data/fixture.fasta \
        --mzml "$work/fixture.mzML" --out-dir "$stub/defer_kept" --config "$stub/defer.json" \
        --threads 2; then
    echo "a deferred-DeepLC run accepted bands that kept the placeholder iRT"; exit 1
fi
grep -q "kept the input iRT" "$stub/defer_kept.log" \
    || { tail -20 "$stub/defer_kept.log"; echo "the refusal did not say why"; exit 1; }
grep -q "reusing a previous run's adapted bands" "$stub/union.log" \
    || { echo "run b of the once_per_run arm did not reuse run a's bands"; exit 1; }
grep -q "the bands keep those values" "$stub/keep.log" \
    || { echo "the keep arm did not keep the experiment-level re-prediction"; exit 1; }
"$PY" - "$stub" "$work" <<'PYEOF'
import json, os, sys
from pathlib import Path
import pyarrow.parquet as pq

stub, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
def calls(name):
    path = stub / (name + ".calls.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
def same(a, b):
    return Path(a).resolve() == Path(b).resolve()
def fail(msg):
    sys.exit("stub arms: " + msg)
bands = ("g00", "g01")

# once_per_run + multi-head: ONE worker call for run a's two bands, written under the names
# run b's shared_bands looks for; run b calls nothing and writes no band table.
c = calls("union")
if len(c) != 1 or c[0]["mode"] != "multihead" or not c[0]["bands"]:
    fail("union: expected one --bands multi-head call, got %r" % c)
want = [(stub / "union/a/groups" / g / "lib_precursors.parquet",
         stub / "union/a/groups" / g / "lib_precursors_multihead.parquet") for g in bands]
if [(Path(i).resolve(), Path(o).resolve()) for i, o in c[0]["pairs"]] != want:
    fail("union: band pairs %r, expected %r" % (c[0]["pairs"], want))
for g in bands:
    if not (stub / "union/a/groups" / g / "lib_precursors_multihead.parquet").exists():
        fail("union: run a wrote no adapted table for " + g)
    if list((stub / "union/b/groups" / g).glob("lib_precursors*.parquet")):
        fail("union: run b wrote a band table for %s although it reuses run a's" % g)

# once_per_run, multi-head off: the experiment re-predicts the library once and the bands
# keep it, so the worker runs once and no band table is written at all.
c = calls("keep")
if len(c) != 1 or c[0]["mode"] != "repredict" or c[0]["bands"]:
    fail("keep: expected one whole-library re-prediction, got %r" % c)
if not same(c[0]["pairs"][0][0], work / "lib_prec.parquet"):
    fail("keep: re-predicted %r, not the imported library" % c[0]["pairs"][0][0])
for run in ("a", "b"):
    for g in bands:
        if list((stub / "keep" / run / "groups" / g).glob("lib_precursors*.parquet")):
            fail("keep: %s/%s wrote a band table" % (run, g))

# per_band, multi-head off: the experiment re-prediction, then one per band per run, and run
# b takes run a's band slices instead of writing its own.
c = calls("perband")
if [x["mode"] for x in c] != ["repredict"] * 5 or any(x["bands"] for x in c):
    fail("perband: expected five single-table re-predictions, got %r" % c)
for k, (run, g) in enumerate([(r, g) for r in ("a", "b") for g in bands], start=1):
    lib_in, lib_out = c[k]["pairs"][0]
    if not same(lib_in, stub / "perband/a/groups" / g / "lib_precursors.parquet"):
        fail("perband: run %s band %s read %r, not run a's slice" % (run, g, lib_in))
    if not same(lib_out, stub / "perband" / run / "groups" / g / "lib_precursors_deeplc.parquet"):
        fail("perband: run %s band %s wrote %r" % (run, g, lib_out))
    if (stub / "perband/b/groups" / g / "lib_precursors.parquet").exists():
        fail("perband: run b wrote its own slice of " + g)

# The overlap: the same single multi-head call on run a, and the same scored table as the
# runs one after the other (the stub makes the adaptation a copy, so the fronts are the
# only thing the overlap could change).
for name in ("seq", "overlap"):
    c = calls(name)
    if len(c) != 1 or c[0]["mode"] != "multihead" or c[0]["bands"]:
        fail("%s: expected one multi-head call, got %r" % (name, c))
    if not same(c[0]["seed"], stub / name / "a/seed_psms.parquet"):
        fail("%s: calibrated against %r" % (name, c[0]["seed"]))
a = pq.read_table(stub / "seq/scored_combined.parquet")
b = pq.read_table(stub / "overlap/scored_combined.parquet")
if not a.equals(b):
    fail("the overlapped experiment scored differently from the sequential one")
for run in ("a", "b"):
    if not (stub / "overlap" / run / "seed_psms.parquet").exists():
        fail("overlap: run %s has no seed table" % run)

# The deferred build: one --bands multi-head call per run, over the two bands.
for name in ("defer_fast", "defer_kept"):
    c = calls(name)
    if len(c) != 1 or c[0]["mode"] != "multihead" or len(c[0]["pairs"]) != 2:
        fail("%s: expected one two-band multi-head call, got %r" % (name, c))
print("    ok: once_per_run union, kept re-prediction, per-band slices reused, deferred "
      "library refused and accepted, overlap equal to sequential (%d stub calls)"
      % sum(len(calls(n)) for n in ("union", "keep", "perband", "seq", "overlap",
                                     "overlap_fail", "defer_fast", "defer_kept")))
PYEOF

# 6. Assertions.
echo "=== smoke: assertions"
#
# --min-assertions is the count cited in CHANGELOG.md and docs/19_getting_started.md.
# It said 112 in both while 117 actually ran, because nothing checked. It also
# catches a guard block that silently stopped executing, which fails no assertion
# and so reads as a pass.
"$PY" ci/check_smoke.py --out-dir "$work/out" --planted "$work/planted.json" \
    --compare-peptides "$work/out2/peptides.tsv" --experiment "$work/exp" \
    --min-assertions 136

echo "=== smoke: also check inspect and report run standalone"
"$BIN" inspect "$work/out/psms_scored.parquet" > /dev/null
"$BIN" report --psms-scored "$work/out/psms_scored.parquet" --out-dir "$work/report_only" \
    --peptide-quant "$work/out/peptide_quant.parquet" --q 0.05 > /dev/null
test -s "$work/report_only/peptides.tsv"

# Hashes of the user-facing outputs, for the cross-platform comparison in CI.
# Measured 2026-08-27: these are IDENTICAL on Windows and Linux, down to the
# quantity digits, so the native pipeline is byte-reproducible across operating
# systems and not merely across runs on one machine. That is a property worth
# keeping, and the only way to keep it is to check it.
#
# No expected value is committed. A golden hash would have to be updated by every
# legitimate change to scoring, which turns an improvement into a chore and
# eventually into a rubber stamp. CI compares the two platforms against each
# other instead.
hashfile="$repo/smoke_output_hashes.txt"
: > "$hashfile"
for f in peptides.tsv proteins.tsv; do
    if command -v sha256sum >/dev/null 2>&1; then
        h=$(sha256sum "$work/out/$f" | cut -d" " -f1)
    else
        h=$(shasum -a 256 "$work/out/$f" | cut -d" " -f1)
    fi
    echo "$h  $f" >> "$hashfile"
done
echo "=== smoke: output hashes"
cat "$hashfile"

echo "SMOKE_OK"
