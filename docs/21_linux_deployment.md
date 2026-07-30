# Linux CPU-only deployment: the Strasbourg prenylation batch

Target: Linux server, **no GPU**, checkout at `/home/robbin/MuMDIA/`, mzML at
`/public/conode53/robbin/mzml`. Written for the 83-file Strasbourg batch; the tuning section
generalises.

The other docs (notably `19_getting_started.md` and `17_troubleshooting.md`) use the Windows
development machine's paths, including advice to "use a native Windows path". Ignore that
here.

---

## 0. What has never run on Linux

The Parquet handoff (`rescore.handoff = parquet`, `mumdia_io::table::BatchWriter`) and the
worker's Parquet reader were developed and measured on Windows only. Arrow/Parquet is
portable and nothing in them is platform-specific, but **run the smoke test in section 5
before committing an 83-file batch to it.** If it misbehaves, `"handoff": "tsv"` restores the
previous path with no other change.

---

## 1. Build

```bash
cd /home/robbin/MuMDIA/rust/mumdia
cargo build --release --locked
./target/release/mumdia --version
```

**Trap.** `rust/mumdia/.cargo/config.toml` sets `target-dir = "C:/Users/robbi/mumdia_build"`.
It is gitignored, so a `git clone` is clean, but an `rsync`/`scp` of the working tree carries
it — and on Linux that string is *relative*, so cargo silently builds into
`rust/mumdia/C:/Users/robbi/mumdia_build/release/` and `target/release/mumdia` never appears.
Deploy by `git clone`, or delete that file, or `export CARGO_TARGET_DIR=/home/robbin/mumdia_build`.

---

## 2. Sidecar environments

Two interpreters are needed. CPU-only torch — do not let pip pull a CUDA build (~2.5 GB of
wasted download on a GPU-less box).

```bash
# rescorer: nn_torch
conda create -y -n mumdia_rescore python=3.11 numpy pandas pyarrow
conda run -n mumdia_rescore pip install torch --index-url https://download.pytorch.org/whl/cpu

# retention time: DeepLC multitask
conda create -y -n deeplc_mt python=3.11 numpy pandas pyarrow
conda run -n deeplc_mt pip install deeplc psm_utils
conda run -n deeplc_mt pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Point the config at their `bin/python` (see `config.strasbourg-linux.json`). Paths must be
**absolute and start with `/`**: a `c:/...` path is relative on Linux, and script resolution
then builds nonsense paths before failing with a message naming a Windows directory.

Verify both before anything long:

```bash
./target/release/mumdia doctor --config /home/robbin/MuMDIA/config.strasbourg-linux.json
```

`doctor` probes every configured interpreter and checks `torch,numpy,pandas,pyarrow` for
`nn_torch`. Preflight now also fails fast if an interpreter path does not exist — previously a
mistyped `rescore.python` was only discovered *after* all 83 per-run chains, discarding the
whole batch's compute at the final stage.

---

## 3. Data to copy

Nothing multi-GB travels with git (`*.parquet`, `mzml_files/`, `fasta/` are all ignored). Put
data outside the checkout so `git pull` never touches it.

```
/home/robbin/MuMDIA/data/lib/      libraries
/home/robbin/MuMDIA/data/fasta/    FASTA
```

**Focused library (recommended, 1.81 GB)** — 13,599,872 precursors / 162,716,592 fragments:

| from | to |
|---|---|
| `H:/strasbourg_out/focused_precursors.parquet` (0.40 GB) | `data/lib/` |
| `H:/strasbourg_out/focused_fragments.parquet` (1.42 GB) | `data/lib/` |

**Full prenyl library (5.27 GB)** — 54,821,556 precursors (27.4M target + 27.4M decoy) /
657,121,084 fragments:

| from | to |
|---|---|
| `H:/strasbourg_out/lib_prenyl_precursors.parquet` (1.36 GB) | `data/lib/` |
| `H:/strasbourg_out/lib_prenyl_fragments.parquet` (3.91 GB) | `data/lib/` |

**Both need** `fasta/human_crap.fasta` (13.1 MB) → `data/fasta/`.

**Do not copy** `lib_prenyl_precursors_ft_full.parquet`: its `predicted_irt` is already
DeepLC-fine-tuned, and fine-tuning on top of a fine-tune is exactly what CLAUDE.md warns
against. The server produces its own. Also skip `np_*` (the non-prenyl control) and `pruned_*`
(a near-duplicate of focused).

---

## 4. Environment variables

```bash
unset OMP_NUM_THREADS              # see trap below
export MUMDIA_NN_THREADS=16        # torch saturates ~16; 32 measured SLOWER
export DEEPLC_FT_THREADS=$(nproc)  # the only handle on the DeepLC phase
export TMPDIR=/home/robbin/tmp     # local disk, not the network mount
```

**`OMP_NUM_THREADS` trap.** `OMP_NUM_THREADS=1` is a common cluster module default. It used to
silently override `MUMDIA_NN_THREADS`, pinning the rescore to one thread with no way to fix it
from MuMDIA's knob. Precedence is now: explicit `MUMDIA_NN_THREADS` wins, `OMP_NUM_THREADS` is
honoured only when the MuMDIA knob is unset. The worker prints which source it used.

**`DEEPLC_FT_THREADS` matters more than it looks.** The Rust caller never passes `--threads`
or `--predict-threads` to `deeplc_finetune.py`, so this variable is the only control, and it
governs the whole-library prediction phase, which dominates that step (30.8 min of a 36.5 min
fine-tune on a 4.9M-peptidoform library). It defaults to 8 regardless of core count.

**Iteration/fold traps.** `MUMDIA_NN_FOLDS`, `MUMDIA_NN_ITERS` and `MUMDIA_NN_TRAIN_FDR` set
in the shell are **silently overridden** by the Rust caller. Change `rescore.folds`,
`rescore.num_iter`, `rescore.train_fdr` in the config instead. `rescore.num_iter` defaults to
10 (the worker's own default of 5 never applies through the engine).

---

## 5. Smoke test before the batch

Two files, not 83. Confirms the whole chain including the never-Linux-tested Parquet path.

```bash
cd /home/robbin/MuMDIA
M=/public/conode53/robbin/mzml
./rust/mumdia/target/release/mumdia run-experiment \
  --lib-precursors data/lib/focused_precursors.parquet \
  --lib-fragments  data/lib/focused_fragments.parquet \
  --mzml $M/<file1>.mzML --mzml $M/<file2>.mzML \
  --run-names r01 --run-names r02 \
  --out-dir /home/robbin/scratch/smoke \
  --config config.strasbourg-linux.json \
  --top-peaks-ms2 300 2>&1 | tee smoke.log
```

Check in `smoke.log`:

- `wrote the sidecar feature table as parquet` — the Rust writer worked
- `format=parquet` from the worker — it read it back
- `torch cpu threads=N (from MUMDIA_NN_THREADS; M cores visible)` — threading as intended
- `rt-im-train: ... status="loess"` for **both** runs, and `reusing this fine-tuned library`
  before run 2 — fine-tune once, calibrate per run
- `rescore: done ... target_peptides_at_1pct=` non-zero, and a decoy fraction near 1%

---

## 6. The batch

`--out-dir` must be on **local disk**. The streaming memmap lands in
`<out-dir>/sidecar_work/`, and serving multi-GB random access over NFS would be painful.
Inputs can stay on `/public/conode53`.

```bash
cd /home/robbin/MuMDIA
M=/public/conode53/robbin/mzml
ARGS=(); i=0
for f in "$M"/*.mzML; do
  i=$((i+1)); ARGS+=(--mzml "$f" --run-names "$(printf 'r%02d' $i)")
done
echo "queued $i runs"
nohup ./rust/mumdia/target/release/mumdia run-experiment \
  --lib-precursors data/lib/focused_precursors.parquet \
  --lib-fragments  data/lib/focused_fragments.parquet \
  "${ARGS[@]}" \
  --out-dir /home/robbin/scratch/strasbourg83 \
  --config config.strasbourg-linux.json \
  --top-peaks-ms2 300 > strasbourg83.log 2>&1 &
```

Run names are `r01..r83` in glob order; the `source` index in `scored_combined.parquet`
follows that order, so keep the mapping (`echo "${ARGS[@]}"`) if you need to trace a run back
to a file.

---

## 7. Memory and `parallel_runs`

`run-experiment` fans out **inside one process**, so the fragment index is shared, not
duplicated per concurrent run — only the per-run working set adds up.

Resident cost is roughly **28 B/fragment** (the `Library` arrays and the `FragIndex` postings
both live during extract), plus ~24 B/candidate, plus hit accumulation, which is the dominant
and least predictable term (it scales with library size × RT window width × scans).

| library | fragments | index floor | measured/expected per-run peak |
|---|---|---|---|
| focused | 162.7M | ~4.6 GB | ~20 GB (measured on this data) |
| full prenyl | 657.1M | ~18.4 GB | expect 60–80 GB |

Guidance, focused library: `parallel_runs=1` under 48 GB, 2 at 64–128 GB, 3–4 above 256 GB.
Full prenyl library: `parallel_runs=1` unless you have ≥256 GB. Start at 1 and raise it after
watching one run's peak RSS — a failure 40 runs in is far more expensive than a slower start.

---

## 8. The rescore at 83 runs

The pooled PSM count will be large (six HYE runs gave 8.86M). Relevant behaviour:

- **`"handoff": "parquet"` is set in the Linux config, and matters most here.** With TSV, six
  runs produced a 30.18 GB text PIN, which exceeded the 4 GB streaming threshold and made every
  self-training iteration re-read a 12.77 GB memmap. Measured: 671.6 min → 12 min, decoy
  fraction unchanged at 0.988%.
- The stream/in-memory decision uses **decoded** bytes for Parquet, so it is not fooled by
  compression. At 83 runs the pool will very likely exceed RAM, and streaming is then correct —
  it is a low-memory guarantee, not a bug.
- If it does fit, `MUMDIA_NN_STREAM=0` forces in-memory, which is where the large win comes
  from. The Parquet reader builds one preallocated float32 array batch-wise; budget roughly
  `rows × features × 4 bytes × 2.5`.
- `sidecar_work/` is **not** cleaned up. Expect tens of GB to remain; delete it after the run.

---

## 9. DeepLC scope

`experiment.finetune_scope = first_run_only` (set in the config): fine-tune once on run 1,
then every run fits its own LOESS calibration against that library.

Measured on the 6-run HYE set: reuse costs **+7.2 s median RT error (+47%)**, windows widen
145 s → 179–227 s, and the degradation is **monotonic in acquisition order** — real
chromatographic drift a single fine-tune cannot track. It cost only **0.02%** of peptides
there, and it turns ~83 × 40 min ≈ 55 h of fine-tuning into ~40 min.

For 83 files the drift will exceed what 6 files showed. If identifications look weak in the
later runs, the answer is periodic re-fine-tuning (every Nth run) rather than either extreme;
that is not implemented yet. `per_run` is available if the hours are affordable.
