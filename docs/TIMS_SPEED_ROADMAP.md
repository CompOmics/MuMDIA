# TIMS speed roadmap: wall time and memory of the best diaPASEF pipeline

Status, 2026-10-06: M0 reviewed; plan approved; steps 1-8 carried out (section 7); summary in section 8.

## 1. Scope

The workload is ProteoBench `quant_lfq_DIA_ion_diaPASEF` (HYE, 3 + 3 runs) with MuMDIA's best
configuration so far. It has two steps, and both are in scope.

1. **Pass 1**, the end-to-end run `e2e_rx2` (config
   `/public/local/ProteoBench/HYE_diaPASEF_mumdia/e2e_rx2/config.json`, binary `~/bin/mumdia-rx/mumdia`,
   `--threads 64`):
   - the `l1d` FASTA library (14,731,188 precursors, 176,774,256 fragments);
   - `library_irt: library` with `multihead_calibration: 80`;
   - `rt_im_train.refit` (pooled pass-1 rescore, per-run refit, second extract to compete);
   - `extract.im_gate: fragments`, `retain_top_peaks: 5`;
   - `retrace` with `repick` and `features.retrace_apex`;
   - Extended plus IM features; pooled `nn_torch` rescore;
   - the diaPASEF quant preset (TIMS_QUANT_ROADMAP 4g).

   Measured before this roadmap: 3:29:14 wall, 135 GB peak RSS, 1,547% CPU on 64 threads. The
   transfer MBR tiers of that run (rescuable and re-extraction) are not part of the target pipeline,
   because the second pass replaces them.
2. **Second pass**, arm D of TIMS_QUANT_ROADMAP 4n, in the engine form that SP6 describes
   (`run-experiment`, opt-in, not built yet):
   - library: pass-1 `precursor_q` <= 0.05, predicted fragment intensities, empirical RT and 1/K0;
   - decoys: new, interior reversed, both termini kept;
   - windows: RT half-width = held-out p99, 1/K0 half-width 0.04;
   - two trace sets: the ID chain on the narrow windows, the quant traces on the pass-1 widths;
   - pooled rescore; report limited to pass-1 `precursor_q` <= 0.01; preset quant.

   Prototype: `quant_diag/sp_lib.py`, `sp_chain.sh`, `sp_eval.sh`, `sp_arms.sh`, `sp_seeds.sh`.

Targets:
- wall time on this host (2x EPYC 9354, 128 CPUs, 755 GB, one 15 TB HDD at `/public/local`, a 3.5 TB
  SSD at `/`);
- peak memory at or below **64 GB** for the whole process tree (decision 2026-10-05, after M0), so
  that a 64 GB node can run the pipeline. The budget counts anonymous memory, plus the file-backed
  matrices that a stage must hold resident to run at speed (the rescore's training matrix).

## 2. Rules and equivalence gate

- No loss of sensitivity or quantification accuracy. About-1% wins that need much code are skipped.
- Clean room: DIA-NN is a reference for methods and results only.
- At most three retraces at a time. Every timed arm sets `MUMDIA_CACHE_DIR=off` or a fresh directory.
- Timed arms start on a quiet host (1-minute load below 10) and record the load.

A change passes the equivalence gate ("within seed noise") when all of these hold:
- On two ID sets, the mean over 3 rescore seeds of ProteoBench ions, global, eq and PB CV, the
  DIA-NN-shared numbers (`DIANN=220 shared.py`) and the per-run accepted rows on `run_psm_q` stay
  inside the current seed spread.
  - Pass 1 reference arms: `eng_repick` and `eng_repick_s1`.
  - Second pass reference arms: `sp_eng_repick_A05pd` (s0, rescore seeds 0-2) and
    `sp_eng_repick_s1_A05pd`.
- A change that alters the candidate set, the features or the training population also needs the
  single-run E. coli entrapment (TIMS_ROADMAP_bis section 8) to show an unchanged FDP.
- Single-seed deltas under about 0.4% are noise (CLAUDE.md).

Each lever states which level it reaches:
- **B (byte-identical):** every artifact the later stages read is bit-identical. No seed runs needed.
- **F (float-equivalent):** values differ in the last bits; identifications within seed noise.
- **S (seed noise):** the equivalence gate above, with entrapment where it applies.

## 3. M0: measurements (2026-10-04/05)

### 3.1 Method

- **Pass 1:** the e2e_rx2 command, rerun unchanged into `HYE_diaPASEF_mumdia/m0_e2e/`.
  - Same binary, config and `--threads 64`, and `MUMDIA_CACHE_DIR=off`.
  - The Python workers are those of commit 3f8516c, the source of `mumdia-rx`. The repository's
    `scripts/` now refuse DeepLC 4.4.0, because the 4.5.0 floor arrived with the merge c57fa73. The
    first attempt failed on that check and is kept in `m0_e2e_fail1/`.
  - The run started on a quiet host (1-minute load 2.7) and had the host to itself.
- **Instrumentation:**
  - `/usr/bin/time -v` inside a systemd user scope.
  - A sampler every 2 s: cgroup CPU and memory (anonymous and file pages separately), per-process RSS
    and CPU, busy threads (above half a CPU in the interval), per-process read and write bytes,
    and disk busy time.
  - Phases are cut at the orchestrator's log markers.
  - py-spy (20 Hz) on every Python worker.
- **Standalone probes** (`m0_probe2/`, `m0_probe3/`) on e2e_rx2's r0 pass-2 inputs: extract,
  retrace and features at 16, 32 and 64 threads; retrace with the `.d` cold and warm; 2 and 3
  concurrent retraces. Every probe output is byte-identical to the pipeline's own artifact, so the
  probes measure the pipeline's work. A first probe set (`m0_probe/`) used the wrong window table and
  is discarded.
- **Profiles:** gdb stack samples every 5 s of a symbolised build of the same source. Its outputs are
  byte-identical to `mumdia-rx`.
- **Arm D:** rerun into `sp_eng_repick_A05pd_m0/` with the 3f8516c scripts, timed in a scope.
- **Scripts:** `m0_sample.py`, `m0_stages.py`, `m0_gdbprof.sh`, `m0_fold.py`, `m0_e2e.sh`,
  `m0_probe*.sh`, `m0_d.sh`. They are in the session scratchpad for now; they move to
  `quant_diag/` when the plan is approved.

**Reproduction:**
- Pass 1 gives the same counts as e2e_rx2: 550,484 target PSMs and 100,634 peptides at 1%, and
  107,113 report precursors.
- Arm D gives the same result as section 4n: 105,078 ions, global 0.137, eq 0.209, PB CV 0.092.

### 3.2 Totals

| | e2e_rx2 (2026-10-03) | M0 rerun |
|---|---|---|
| wall | 3:29:14 | **2:53:22** |
| CPU | 1,547% | 1,841% (mean 18.4 of 64 cores) |
| largest single process (`time` max RSS) | 135 GB | 132.7 GB |
| peak anonymous memory of the whole tree (cgroup) | | **113.4 GiB** |
| bytes written | 0.82 TB | 0.82 TB (`time`); 897 GiB (sampler) |
| artifacts kept | 507 GB | 507 GB |

- The 36 minutes between the two walls are host load. The e2e_rx2 run shared the host: its rescore
  workers took 1,376 and 1,172 s against 804 and 846 s here. Only M0 numbers are used below.
- `/usr/bin/time` "File system outputs" counts 512-byte blocks: 1,598,349,536 blocks = 0.82 TB, not
  1.6 TB.
- Per-process RSS sums double-count the rescore worker's shared pages (they reach 257 GB), so memory
  below is the cgroup's anonymous memory unless stated otherwise.

### 3.3 Where the wall time goes (pass 1, M0)

| phase | wall | mean cores | peak anon | note |
|---|---|---|---|---|
| convert, 6 `.d` | 1:24 | 29.5 | 24.8 GiB | 4 at a time |
| seed, 6 runs | 1:46 | 31.6 | 11.3 GiB | |
| multi-head calibration, r0 | 3:20 | 24.6 | 15.2 GiB | DeepLC, 7 shard processes |
| pass-1 chain r0, alone, 64 threads | 8:58 | | 31.8 GiB | extract 4:45 at 8.8 cores, retrace 2:40 at 48, features 1:30 at 10 |
| pass-1 chains r1-r5, 4 at a time, 16 threads each | 32:41 | 29.7 | **113.4 GiB** | r1-r4 22:20; then r5 runs alone on 16 threads for about 10 min |
| pooled pass-1 rescore | 18:18 | | 17.6 GiB + 60 GB file-backed | handoff 3:55 at 4.8 cores; worker 13:24; q and write 1:00 |
| refit and pass 2, **serial**, 64 threads | 67:41 | | 31.7 GiB | 2 more multi-head calibrations on r0 (7:43); per run: refit 4-20 s, extract 5:05-5:23 at 8 cores, retrace 2:44-3:00 at 46, features 1:36-1:45 at 9, compete 3 s |
| pooled pass-2 rescore | 17:15 | | 17.6 GiB + 60 GB file-backed | handoff 2:22; worker 14:06 |
| MBR transfer, re-extraction, quant, report | 21:59 | 1.4-9.3 | 41.0 GiB | not part of the target pipeline, except the pass-1 quant |
| **total** | **2:53:22** | 18.4 | 113.4 GiB | |

The pass-1 quant that arm D needs (fragment areas, one pass) is about 5:30 within the 21:59. The
target pipeline is therefore pass 1 up to the pass-2 rescore (2:31:23) + pass-1 quant (about 5:30) +
second pass (12:15 in the prototype) = **about 2:49**.

**Measured (pass-1 chains of 4):** pass-1 extract 6:43-7:00, retrace 8:20-9:30, and features
2:00-5:40 when four chains run at once. Features alone takes 1:30.

### 3.4 Second pass, arm D (prototype, M0 rerun)

12:15 wall, 31.8 GB peak (largest process), 2,256% CPU.

| step | wall | peak per process | note |
|---|---|---|---|
| library build (`sp_lib.py`, Python) | 1:13 | 7 GB (section 4n) | |
| per-run chains, 3 at a time (6 runs) | 4:20 | | |
| extract (ID windows) | 10-12 s | 9-10 GB | |
| retrace with repick | 8-16 s, or **134-146 s** | 25-27 GB | 134-146 s when the `.d` is not in page cache (r3, r5) |
| features, compete | 3 s | 2 GB | |
| extract and retrace of the quant traces | 30-41 s | 10 GB / 28-30 GB | |
| pooled rescore (936,930 PSMs) | 4:02 | 5.4 GB | |
| evaluation and quant (6 runs in parallel, two passes) | 2:30 | | |

The 86-92 s extract of section 4n is not reproduced: extract takes 10-12 s per run. The second pass
is 7% of the target pipeline. Its slowest part is the cold read of the raw frames from the HDD.

### 3.5 Stage probes (r0 pass-2 inputs; byte-identical outputs)

| stage | 16 threads | 32 | 64 | other arms |
|---|---|---|---|---|
| extract | 5:22 | 4:46 | 4:37 | codec pool 32 or 64: 4:40, 4:42 |
| retrace with repick (`.d` cold) | 8:40 | 5:11 | 3:37 | `.d` warm 3:14; warm + codec pool 32: **2:35**; 2 or 3 concurrent at 16 threads: 8:36, 8:31 |
| features | 1:39 | | 1:24 | codec pool 32: 1:31 |

**Extract is one writer thread.** All 24 windows are probed in the first 20 s at about 45 cores.
The remaining 255 s is the chromatogram writer: `writer_busy_ms=251393` of `elapsed_ms=276588`, and
the probing tasks wait `send_blocked_ms=211766` for it. CPU is about 6 cores and the disk is about
20% busy. The writer encodes one row group at a time and spreads only that group's columns over the
codec pool (`mumdia-io/src/codec.rs:284`). The chromatogram table has few large list columns, so a
larger pool does not help. Retrace writes the same kind of table with 16 row groups in flight.
- The extract table written here is `chromatograms.centroid.parquet` (12 GB, v1 forced under
  retrace, `run.rs:1082-1087`). Only retrace reads it, and in the transfer MBR the re-extraction
  prep.

**Retrace scales with threads and with concurrency up to 48 threads.**
- Three concurrent retraces at 16 threads each take as long as one alone.
- The profile: `visit_events` (sum and joint score) 52% of busy samples and `memcpy` 33%, with 50
  of 64 threads busy.
- A cold `.d` costs 23 s on a quiet HDD and 2 min or more under concurrent I/O (arm D r3/r5,
  e2e_rx2 pass 2).
- The codec pool (default `min(N, 8)`) limits the output write: 32 codec threads take 39 s off,
  byte-identical.

**Features neither scales nor benefits from the codec pool.** About 9 threads are busy, mostly in
parquet decode (`mumdia-io/src/table.rs:2724-2743`) and allocation.

**Quant** (in the pipeline): 18 serial calls of 21-29 s each, at 2-3 cores. The first two calls
took 113 s and 81 s because they read the 17 GB `chromatograms.parquet` cold.

### 3.6 Python workers (py-spy)

- **Multi-head calibration:** 3 calls on the same 14.7M-row library (pass 1, refit f0, refit f1):
  200, 222 and 241 s at about 24 cores.
  - Most of the time is in 7 DeepLC shard processes, 43-48% of it in the convolution forward pass.
  - The projection cache that would let calls 2 and 3 reuse the trunk projections needs DeepLC
    >= 4.5.0. `mumdia-tims` has 4.4.0.
- **NN rescore worker:** 774 s and 814 s.
  - Load and standardise: 71-83 s.
  - Fold training: 703-731 s for 3 folds in parallel, but 330 process-seconds per fold. The folds
    therefore overlap badly.
  - Before training, the worker writes a standardised copy of the whole matrix to a memmap file
    (about 60 GB) so the parallel folds can share it (`nn_rescore_worker.py:1915`). With the engine's
    60 GB handoff, each rescore writes about 124 GB.
  - py-spy sampled only the parent, so the per-fold breakdown is from the worker's own timers.

### 3.7 Idle threads

The run averages 18.4 of 64 cores. The largest idle blocks (core-minutes not used):

| block | wall | cores used | idle core-min |
|---|---|---|---|
| refit pass 2 serial: extract writer (6 runs) | 31 min | 8 | ~1,740 |
| rescore workers, both passes | 27 min | 16 (3 folds x 16 torch threads, poor overlap) | ~1,300 |
| refit pass 2 serial: features (6 runs) | 10 min | 9 | ~550 |
| pass-1 chains of 4: extract writers, features | ~15 min | ~30 | ~500 |
| r5 alone on 16 threads at the end of pass 1 | ~10 min | ~16 | ~480 |
| multi-head calibration x3 | 11 min | 24 | ~440 |
| rescore handoff, both passes (single-threaded stream) | 6 min | 5-8 | ~340 |
| pass-1 quant, 6 serial calls | 5.5 min | 2-3 | ~340 |

### 3.8 Memory

| state | peak anonymous memory |
|---|---|
| 4 pass-1 chains at once (each holding the raw frames during retrace, 33-37 GB) | **113.4 GiB** |
| one chain (extract 30-34 GB, retrace 35-39 GB) | 30-35 GiB |
| rescore worker | 17.6 GiB, plus the 60 GB memmap matrix and the 60 GB handoff, which are file pages |
| multi-head calibration | 15 GiB |
| arm D: 3 chains at once / pooled rescore | 31.8 GB per process / 5.4 GB |

Only the 4-chain phase is above 40 GiB. The rescore needs its 60 GB matrix resident to train
at full speed, even though the matrix is not anonymous memory.

### 3.9 I/O

- **Writes by stage (sampler, GiB):**
  - pass-1 chains 192 (four runs plus r5) and pass-2 chains 259 (extract 86, retrace 120, features
    53);
  - rescore 386 (handoff about 62 and worker memmap about 62 per pass, plus the scored tables);
  - convert 20; everything else under 12 each.
- **Kept, 507 GB.** Of this, 182 GB is never read again after pass-1 features:
  `pass1/chromatograms.parquet` (108 GB) and `pass1/chromatograms.centroid.parquet` (74 GB).
  Without the transfer MBR, the 72 GB pass-2 centroid tables are also dead after retrace.
  `features.parquet` and `psms_competed.parquet` are hard links (compete removes no row), so they
  cost 50 GB per pass once, not twice.
- **Decodes:** each `.d` is decoded 4 times (convert, pass-1 retrace, pass-2 retrace, MBR
  re-extraction retrace), with no frame cache. The spectra parquet is decoded 3 times (seed, extract
  p1, extract p2), and the 14.7M library is loaded for every extract.
- **Disk:** all inputs and outputs share one HDD (`/public/local`, sda). It was 20-80% busy during
  the stages.

## 4. Consequences of the 64 GB budget

- **One per-run chain at a time.** One chain peaks at 30-39 GB (retrace holds the raw frames). Two
  chains at once need about 70-78 GB. Under the budget, the per-run chains therefore run one at a time
  on all 64 threads, and wall time then depends on how well one chain scales with threads. Running
  chains concurrently remains a lever for large hosts; the scheduler sizes it from the measured
  first-chain peak (`sched::bound_by_measured_peak`), and it must learn the budget.
- **Convert at most 2 at a time.** One conversion peaks at 16.5 GB; four at once reached 50.8 GB RSS.
- **The rescore matrix must shrink.** The pooled matrix is 41.6M PSMs x 386 features x 4 B = 60 GB, and
  the worker needs it resident, plus 17.6 GiB of its own. This is the only stage that cannot fit the
  budget by scheduling.
- Arm D (31.8 GB with three chains) and everything else already fit.

## 5. Levers, ranked

Savings are estimated from M0 for this host (64 threads, chains concurrent where memory allows) and,
where it differs, under the budget (chains one at a time). "Level" is the equivalence level of
section 2. The baseline is the target pipeline, about 2:49 (section 3.3).

| # | lever | saving, estimate | memory | level | effort | risk |
|---|---|---|---|---|---|---|
| L1 | extract: encode chromatogram row groups in parallel | 25-35 min (12 extracts: 280 s -> 40-60 s) | + a few row groups in flight | B | medium, engine | low |
| L2 | refit pass 2: run the runs concurrently after r0, and a memory budget in the scheduler | 20-30 min on this host; 0 under the budget | per chain 30-39 GB | B | low, engine | low |
| L3 | rescore: shrink the matrix to fit 64 GB (no standardised copy; 16-bit storage or a projected feature list) | 2-4 min (60 GB less written); needed for the budget | -30 to -60 GB | B for the copy, S for the storage | medium, worker | medium |
| L4 | rescore: fix the fold overlap (703 s wall for 330 s per fold) | 5-10 min | none | B to F | medium, worker | medium |
| L5 | features: parallel decode of the chromatogram input | 6-10 min (12 calls, 90-105 s at 9 cores) | small | B | medium, engine | low |
| L6 | retrace: larger codec pool for the output (measured -39 s) | 5-8 min | none | B (measured) | trivial, engine | none |
| L7 | multi-head calibration: DeepLC 4.5.0 projection cache for the 3 calls on one library | 5.9 min (measured: refit calls 3:42 + 4:01 -> 0:54 + 0:54) | none | S (the 4.5.0 upgrade moves RT; the cache itself is F) | none (installed) | medium |
| L8 | quant: run the per-run calls concurrently | 3-4 min (pass-1 quant 5:30) | 0.3-0.5 GB per call | B | low, engine | none |
| L9 | retrace: read the `.d` ahead, sequentially | 1-3 min quiet; 2 min per run under contention | none | B | low, engine | none |
| L10 | pass-1 scheduling: 6 runs on 4 slots leaves r5 alone on 16 threads | 5-8 min on this host; 0 under the budget | | B | low, engine | none |
| L11 | do not write or keep dead artifacts (pass-1 chromatograms and centroid tables, pass-2 centroid without transfer MBR) | disk 182-254 GB; time only with L1 | none | B | low, engine | none |
| L12 | rescore handoff: stream the features with more than one thread | 3-5 min (3:55 and 2:22 at 5-8 cores) | none | B | low-medium, engine | low |
| L13 | second pass in the engine (SP6), built with L2, L8, L9 from the start | arm D 12 min -> about 6 min | fits | B against the prototype | high (planned anyway) | medium |

Expected total on this host: from about 2:49 to about 1:15-1:30. Under the 64 GB budget (chains one at
a time, L2 and L10 inactive): about 1:45-2:00.

Not taken, with the reason:
- **Rescore recipe knobs that CLAUDE.md measured** (`folds: 2`, `train_subsample`, a larger batch): each
  fails the equivalence gate on at least one HYE pool (`folds: 2` -1.1% on B01). `train_neg_ratio: 2`
  is already the default. The `compact` preset was selected on a DIA-NN library and is
  classifier-specific; it enters only as one option of L3, re-derived for this library and gated.
- **Building the pass-2 library in Rust:** `sp_lib.py` takes 73 s.
- **Reusing decoded spectra, mass calibration and the library between stages:** spectra decode in
  1.4 + 2.3 s, the library loads in 1.9 s, and the mass calibration is a JSON file. Together they
  are under 10 s per extract.
- **A decoded-frame cache shared by the retraces:** the decoded events are 33-39 GB per run against a
  7 GB `.d`. Writing and reading them costs more than decoding again. L9 addresses the part that is
  slow (cold random reads).
- **Convert:** 1:24 for six runs.
- **Scratch on the SSD or tmpfs:** host-specific. The one SSD probe was slower (4:59 against
  3:14), probably because `/` is shared with other users, so it is not a reliable lever here.
- **Streaming retrace over RT slabs** (to fit two chains in 64 GB): high effort. Under the budget one
  chain fits, so it would buy concurrency only. It is parked until L1, L5 and L6 show how well one
  chain scales.

### 5.1 Each lever: measurement and gate

- **L1** Encode the extract chromatograms with the row-group-parallel writer that retrace uses,
  keeping the same row-group boundaries.
  - Measurement: `writer_busy_ms` and `send_blocked_ms` fall; extract r0 at 64, 32 and 16 threads,
    as in section 3.5.
  - Gate: `psms_extracted`, `.peaks` and `chromatograms.centroid.parquet` byte-identical to M0 (`cmp`),
    and peak RSS within 2 GB of M0.
- **L2** Keep r0's refit first (its library is shared). Then run r1-r5 through `plan.map_pooled`, as
  pass 1 does. Add `experiment.memory_budget_gb`, which caps the measured-peak sizing.
  - Measurement: e2e wall, and the peak at budget 64 against no budget.
  - Gate: every per-run artifact byte-identical to M0; peak at most 64 GB with the budget.
- **L3** (a) The worker standardises the engine's raw handoff in place instead of writing a second
  memmap. (b) The matrix is stored as 16-bit, or projected to a feature list re-derived on this
  library and classifier.
  - Measurement: rescore peak (anonymous plus resident matrix) and wall.
  - Gate: (a) scores byte-identical. (b) The full equivalence gate on two ID sets with 3 seeds, plus
    entrapment (the features change).
- **L4** First profile the fold children: py-spy did not follow them in M0; attach to each child by
  PID.
  - Candidate causes: uneven fold sizes, the per-iteration pool scoring (285 process-seconds), and
    memory bandwidth on the shared memmap.
  - Gate: scores byte-identical if only scheduling changes; otherwise the equivalence gate.
- **L5** Profile first (decode in `table.rs:2724-2743`, allocation), then decode the chromatogram row
  groups in parallel.
  - Measurement: features r0 at 64 and 16 threads.
  - Gate: `features.parquet` byte-identical.
- **L6** Retrace's writer takes a codec pool of its own (measured: 32 threads, -39 s).
  - Measurement: retrace r0 with codec pools of 16, 32 and 48.
  - Gate: byte-identical (already shown for 32).
- **L7** Measured in `m0_mh/` (r0, the three calls of M0):
  - Times: without cache 3:16; with a fresh cache 3:14 (miss), 0:54 and 0:54 (hits, which evaluate
    the 80 heads in 26 s). The process peaks at 4.9 GB either way.
  - **The cache is float-equivalent.** Refit f0 with and without cache: max |dRT| 0.012 s, same best
    head (2449).
  - **The upgrade itself is not.** DeepLC 4.5.0 with the current `deeplc_finetune.py`, against 4.4.0
    with the 3f8516c script, on the same anchors:
    - pass-1 call: median |dRT| 0.91 s, p99 20 s;
    - refit f0: median 0.30 s, p99 4,432 s, with 562,896 of 14,731,188 rows (3.8%) more than 60 s
      apart;
    - refit f1: p99 9,310 s.
    The large differences are in the tail that TIMS_ROADMAP_bis calls absurd iRTs (1.95% of rows).
    Which version is closer to the observed RT is not measured.
  - Since 2026-10-05, `mumdia-tims` has DeepLC 4.5.0, so every new e2e runs the new arm.
  - Gate: the full equivalence gate on two ID sets, plus the E. coli entrapment (the RT windows
    change). The first e2e of section 6 step 1 is that measurement. The repository's scripts no
    longer run with 4.4.0, so reverting means pinning the 3f8516c scripts.
- **L8** Run the six per-run quant calls of each quant pass concurrently, each on `threads / 6`.
  - Gate: quant tables byte-identical to M0.
- **L9** Retrace calls `posix_fadvise(WILLNEED)` on `analysis.tdf_bin`, or runs one sequential read
  thread over it, before decoding frames.
  - Prototype: pre-read the `.d` with `cat` in the harness.
  - Measurement: retrace with a cold `.d`, alone and three at once.
  - Gate: byte-identical.
- **L10** When `n_runs` is not a multiple of the run slots, pick slots so that the rounds are even (6
  runs: 3 x 21 threads, or 2 x 32). The alternative is to give the last chain the threads that the
  finished chains release.
  - Measurement: pass-1 phase wall.
  - Gate: byte-identical.
- **L11** Delete `pass1/chromatograms*.parquet` once pass-1 features is written. Do not write the
  pass-2 centroid table to disk when no transfer MBR follows (with L1, retrace can take the rows from
  memory, or the file goes to a deleted temporary).
  - Gate: the remaining artifacts byte-identical; a flag keeps the files for debugging.
- **L12** Measure where the 3:55 and 2:22 go (`features_ms` 154 s, `handoff_encode_ms` 132 s in the
  e2e_rx2 log), then parallelise the slower half.
  - Gate: the handoff file byte-identical.
- **L13** The engine form of SP6:
  - per-run chains under the L2 scheduler, with the `.d` read ahead (L9);
  - the ID traces and the quant traces built in one retrace call with two window sets, which saves
    one frame decode per run;
  - quant per run in parallel (L8);
  - the transfer tiers skipped when `second_pass` is selected.
  - Gate: report and quant tables equal to the prototype on `eng_repick` s0 (B for every artifact
    whose inputs are equal), then the equivalence gate against `sp_eng_repick_A05pd` s0 seeds 0-2 and
    `sp_eng_repick_s1_A05pd`.

## 6. Order of work

Prototype first where the lever needs no code; each step is measured on the full e2e before the next.

1. **Free, no code** (one e2e, about 3 h, plus `eng_repick_s1`-style seeds and entrapment for L7):
   DeepLC 4.5.0 with the projection cache (L7; level S, so it is gated on its own before anything is
   stacked on it), and `MUMDIA_PARQUET_THREADS=32` for the whole run (L6 prototype; check that extract and
   features stay byte-identical, which M0 already showed for one run). The `.d` pre-read in the
   harness (L9 prototype).
2. **L1** (extract writer): the largest single saving, B level.
3. **L2 + L10** (scheduling and the memory budget): small engine changes that use the existing
   scheduler. After this the budget can be enforced and measured.
4. **L6, L8, L9, L11** (small engine changes, B level), as one PR.
5. **L5, L12** (features decode, rescore handoff), after a profile of each.
6. **L3** (rescore memory): (a) first, B level; then (b) with seeds and entrapment. This is the step
   that brings the whole pipeline under 64 GB.
7. **L4** (fold overlap), after the profile of the fold children.
8. **L13** (second pass in the engine), on the scheduler and readers of steps 2-4.

Engine code: L1, L2, L5, L6, L8-L13. Worker code (Python): L3, L4. None: L7.

## 7. Progress

### 7.1 Step 1: L7 gated (2026-10-05)

Arms (harness `quant_diag/m0/s1_l7.sh`): the target pipeline (e2e_rx2 config, `mbr.strategy: none`)
under DeepLC 4.5.0 with the repository's scripts and a fresh projection cache (`s1_dl45/`), against
the M0 run under 4.4.0. Both are rescored with seeds 0-2 on their own pass-2 competed tables and
requantified by the same harness (preset, two passes, `q_filter: psm_q`).

| | 4.4.0 (M0), seeds 0 / 1 / 2 | 4.5.0 + cache, seeds 0 / 1 / 2 |
|---|---|---|
| ProteoBench ions | 93,743 / 93,693 / 93,718 | 94,151 / 94,513 / 94,260 |
| global | 0.131 / 0.131 / 0.131 | 0.132 / 0.132 / 0.132 |
| eq | 0.179 / 0.180 / 0.179 | 0.180 / 0.181 / 0.180 |
| PB CV | 0.084 / 0.084 / 0.084 | 0.084 / 0.085 / 0.084 |
| DIA-NN-shared ions | 87,527 / 87,511 / 87,530 | 87,918 / 88,178 / 87,982 |
| shared: eps / CV / E. coli log2 | 0.128 / 0.086 / -1.84 | 0.128 / 0.086 / -1.84 |
| per-run rows on `run_psm_q`, sum | 550,554 / 550,104 / 550,207 | 552,851 / 554,895 / 553,747 |
| E. coli entrapment: real peptides | 13,906 / 13,948 / 14,071 (`MuMDIA_repick/full`) | 13,995 / 13,950 / 13,918 |
| E. coli entrapment: FDP | 0.473% / 0.367% / 0.432% | 0.402% / 0.451% / 0.416% |

- **Sensitivity:** ions +0.6%, DIA-NN-shared ions +0.6%, and per-run rows +0.6%, each outside the
  4.4.0 seed spread and in the gaining direction.
- **Accuracy:** global and eq are +0.001 (eq 0.180-0.181 against 0.179-0.180). This is the size of the
  rounding, and below the 0.002-0.003 that separated the arms of TIMS_QUANT_ROADMAP 4n. PB CV and the
  shared-ion numbers are unchanged.
- **FDR:** the entrapment FDP stays inside the 4.4.0 spread.
- **Decision:** L7 is taken, at level S. `mumdia-tims` stays on DeepLC 4.5.0, and the 4.5.0 arm is
  the reference for the following steps.
- **Cost:** the two refit calibrations take 61 s and 75 s against 222 s and 241 s (-5.5 min). The
  pass-1 call is a cache miss (3:33). The whole run took 2:40:00, but convert (4:06 against 1:24, the
  `.d` files were cold) and both rescore workers (943 and 937 s against 804 and 846 s) were slower than
  in M0. A single e2e wall time carries several minutes of host variance, so levers are judged on
  their phases.

### 7.2 First speed batch: L1, L2, L4, L6, L8, L10, L11 (2026-10-05)

Engine changes in the working tree (HEAD), fmt, clippy and the workspace tests clean (565 tests in
`mumdia`). The validation binary `~/bin/mumdia-sp1` is the same patch on 3f8516c, so that its artifacts can
be compared byte for byte with the L7 arm, which ran on that source; L4 is a change to
`scripts/nn_rescore_worker.py`. Harness `quant_diag/m0/v_sp1.sh`.

- **L1** (`mumdia-io` `ColumnEncoder::with_row_groups_in_flight`, 16 for extract's chromatograms):
  - extract r0 at 64 threads 1:23 against 4:37; at 16 threads 2:11 against 5:22;
  - writer busy 57 s against 251 s; peak unchanged (33 GB);
  - `psms_extracted`, `.peaks` and the centroid table byte-identical to the old binary at the same
    thread count;
  - the unit test pins byte identity with the serial writer for 2 and 16 groups in flight.
  - The old binary's centroid table already differs in layout between 16 and 64 threads (same rows,
    `cmp_content.py`), which is the documented `--threads` behaviour.
- **L2 + L10:** the refit's pass 2 runs r0 alone, then the other runs under the measured-peak plan,
  in even rounds. `experiment.memory_budget_gb` (default 0) caps the plans.
- **L4:** the worker no longer flushes the shared memmap before the fold children start.
- **L6:** codec pool cap 8 -> 32. The retrace gain is being re-measured (two warm retraces with the
  same settings took 3:14 and 2:38).
- **L8:** the per-run quant calls of each quant pass run at once.
- **L11:** under the refit, pass 1's trace tables are deleted after the pooled pass-1 rescore, and the
  pass-2 centroid table after retrace unless `mbr.reextract` reads it (`MUMDIA_KEEP_INTERMEDIATE=1`
  keeps them).

The target pipeline, same inputs and config as the L7 arm:

| | L7 arm (`s1_dl45`) | batch (`v_sp1`) |
|---|---|---|
| wall | 2:40:00 | **1:46:20** (-34%) |
| mean cores used | 18.8 | 27.7 |
| pass-1 chains (r1-r5) | 32:41 (4 x 16 threads, then r5 alone) | 27:18 (3 x 21 threads, two rounds) |
| rescore worker, pass 1 / pass 2 | 943 / 937 s | 454 / 492 s |
| refit and pass 2 | 63:11 (serial) | 34:36 (r0, then 3 x 21 threads) |
| quant, two passes | about 7:30 | about 3:30 |
| largest process (`time`) | 130.4 GB | 130.4 GB |
| peak anonymous memory | | 89.2 GiB (three chains at about 39 GB) |

- **Equivalence: level B.** All 169 tables that both runs keep are byte-identical: every scored
  table, `peptide_quant`, `protein_group_quant`, `fragment_quant`, the LFQ matrices and the TSV
  reports. The 24 files missing are the deleted intermediates (L11). No seed or entrapment run is
  needed for this batch.
- **One regression to watch:** the pass-1 rescore handoff took 347 s against 145 s. It now starts
  while the disk still writes back the pass-1 chains' output.

**With the 64 GB budget** (`v_sp1_b64`, `experiment.memory_budget_gb: 64`, same binary):
- **Wall:** 1:51:54, 5.5 min more than without a budget.
- **Equivalence:** the same 169 tables are byte-identical to the L7 arm.
- **Plans the budget chose:** conversions and seeds 4 at a time; every per-run chain alone on 64
  threads (`chains_that_fit=1`, from a measured chain peak of 39 GB).
- **Peak anonymous memory:** 32.6 GiB, against 89.2 GiB without a budget.
- **The rescore is the one remaining excess.** Its largest process still reaches 130 GB, all of it
  page-cache mappings of the 60 GB raw handoff and the 60 GB standardised matrix, of which training
  needs one resident. L3 (float16 storage) is what brings this under 64 GB.

After L1, one chain on 64 threads is about as fast as three on 21 each, so under the budget
running the chains one at a time costs little.

### 7.3 Measured and dropped: L6 and L14; L9 and L3 in progress (2026-10-05)

- **L6 is noise, reverted.** Warm retrace r0 on 64 threads, alternating, three repeats each
  (`p_l6/`): codec pool 8: 3:02, 2:54, 2:37; codec pool 32: 2:57, 2:37, 2:50. A warm retrace varies
  by about +-12 s between identical runs, and the 39 s of section 3.5 was one draw. The cap stays 8.
- **L14 (new, dropped): a scan-block index for retrace's event lookups.** Each query walked every
  event of a TOF window over all ~900 scans of a frame and kept only the scans of the 1/K0 band. The
  index (blocks of 16 scans, TOF-sorted, matched events re-sorted into the old order) visited the same
  events in the same order and was byte-identical (`p_l14/`, unit test). But retrace's
  sum-and-write phase took 120-129 s with and without it (warm, 64 threads). The visitor is busy in
  the profile but does not bound that phase's wall time; the read, sum and write pipeline does.
  Reverted: code without a gain.
- **The retrace profile, read again.** A third of retrace's busy samples were `memcpy` inside the
  zstd frame decoder, reading the memory-mapped `analysis.tdf_bin`. On a warm `.d` the frame decode
  takes 3-5 s, and on a cold one 61 s. Those samples are page faults on a cold file, not copy work.
  This is what L9 addresses.
- **L9, in the working tree, validation pending:** a thread reads `analysis.tdf_bin` sequentially
  when extract starts (34 s cold on this HDD, against 61 s of cold random reads in retrace), so that
  retrace finds it warm. It changes no output; `MUMDIA_RETRACE_PREREAD=0` turns it off. Its effect
  is expected where retrace reads a cold file: pass 2 of the refit, and every chain on a node whose
  page cache cannot hold six `.d` files.
- **L3, functional test** (arm D's pooled rescore, 936,930 PSMs, one seed, `t_l3/`):
  - `MUMDIA_NN_STORE=f32` (the default) is byte-identical to the previous worker.
  - `f16`: 666,125 against 667,400 target PSMs at 1% (-0.19%); 115,363 against 115,380 peptides
    (-0.01%); worker peak 4.5 against 5.4 GB.
  - Means and standard deviations are the float32 path's, bit for bit (same moment partition), and
    every stored value is the float32 value rounded to float16.
  - The full gate (two ID sets, seeds 0-2, entrapment) is running (`quant_diag/m0/s3_l3.sh`).

### 7.4 L3 gated; L9 dropped (2026-10-06)

**L3, `MUMDIA_NN_STORE=f16`** (opt-in in `scripts/nn_rescore_worker.py`; `quant_diag/m0/s3_l3.sh`,
`s3_entrap.sh`). Rescore seeds 0-2, the same requant and scoring as step 1.

| | float32, seeds 0 / 1 / 2 | float16, seeds 0 / 1 / 2 |
|---|---|---|
| A (target pipeline, pass-2 competed): ions | 94,151 / 94,513 / 94,260 | 94,210 / 94,337 / 94,194 |
| A: global / eq / PB CV | 0.132 / 0.180-0.181 / 0.084-0.085 | 0.132 / 0.179-0.180 / 0.084 |
| A: DIA-NN-shared ions | 87,918 / 88,178 / 87,982 | 87,846 / 87,955 / 87,887 |
| A: per-run rows on `run_psm_q`, sum | 552,851 / 554,895 / 553,747 | 553,160 / 554,205 / 553,384 |
| B (`eng_repick` competed): ions | 92,658 / 92,707 / 92,174 | 92,509 / 92,452 / 92,459 |
| B: global / eq / PB CV | 0.130 / 0.177-0.178 / 0.083 | 0.130 / 0.177-0.178 / 0.083 |
| B: DIA-NN-shared ions | 86,361 / 86,364 / 85,937 | 86,250 / 86,194 / 86,238 |
| B: per-run rows, sum | 543,262 / 543,781 / 541,021 | 542,984 / 542,463 / 542,452 |
| E. coli entrapment: real peptides | 13,995 / 13,950 / 13,918 | 14,013 / 13,958 / 13,969 |
| E. coli entrapment: FDP | 0.402% / 0.451% / 0.416% | 0.470% / 0.419% / 0.415% |
| largest rescore process (HYE) | 133.7-133.9 GB | 98.5-102.0 GB |

- **Means inside the float32 seed spread:** ions, global, eq, PB CV and per-run rows, on both ID sets.
- **DIA-NN-shared ions** on set A are the closest call: mean 87,896 against 88,026 (-0.15%), with
  two of three float16 seeds just below the float32 minimum. Set B's shared ions are inside the
  spread.
- **FDR:** the entrapment FDP stays inside the range measured for pass 1 so far (0.367-0.473%), and
  real peptides +0.2%.
- **Level S, passed.**
- The first entrapment attempt used the older entrapment binary, whose parquet handoff makes the
  worker fall back to float32 (it logs this); the matched pair above uses `mumdia-rx` (raw handoff)
  for both arms.
- The largest-process numbers count page-cache mappings: the 60 GB raw handoff is mapped while the
  float16 matrix is filled. The test that matters for a 64 GB node is a run under a 64 GB cgroup
  limit (in progress).

**L9 dropped** (`v_sp3_b64`: the batch plus L9, at the budget; byte-identical, 169 tables).
- Pass-2 retraces: 1,053 s against 1,116 s in total (-63 s).
- Pass-2 extracts: 668 s against 526 s (+142 s). The read-ahead shares the HDD with extract's reads
  and writes, and on this host most `.d` pages were still cached.
- The whole run took 2:04:39 against 1:51:54, mostly in convert (442 s against 199 s, cold `.d`) and
  the pass-1 rescore, which L9 does not touch.
- Reverted. A cold `.d` remains a cost on nodes whose page cache cannot hold the run's raw files;
  the fix there is a faster disk, not a read-ahead competing for the same one.

**L3 under a 64 GB cgroup limit** (`s3_cap64/`: ID set A, seed 0, `systemd-run -p MemoryMax=64G
-p MemorySwapMax=0`, page cache included in the limit).

| | uncapped | capped at 64 GB |
|---|---|---|
| float16 | 764 s | **2,147 s**, byte-identical scores |
| float32 | 700-1,656 s | not finished in 45 min (stopped) |

- With float16 the pooled rescore of the target pipeline runs on 64 GB. Fold training is unchanged
  (495 s). The load phase took 978 s against about 80 s: the worker reads the 60 GB raw handoff
  twice (moments, then standardise), and under the limit the page cache cannot keep it between the
  passes, so the second pass reads it from the HDD again. On an SSD node this costs about a minute.
- Follow-up L3c: the engine computes the per-column moments as it writes the raw handoff, so the
  worker needs one pass. Level F at best, because the float64 sums would have to be accumulated
  in the worker's partition and order to stay bit-identical.
- **Decision (2026-10-06): `MUMDIA_NN_STORE=f16` stays opt-in.** It is not the default and is not
  implied by `experiment.memory_budget_gb`. A run on a 64 GB node sets it explicitly.

### 7.5 L5 taken; L12 parked (2026-10-06)

**L5: `features.chrom_loaders` default 3 -> 5.** Features r0 (pass 2), 64 threads, alternating,
two repeats each (`p_l5/`):
- 3 loaders: 1:30 and 1:30, main pass bound by the loaders (`binding="loader"`, 3 loaders busy
  190 s, compute 10 s);
- 5 loaders: 1:12 and 1:00, RSS 7.0-7.1 against 6.0-6.2 GB;
- `features.parquet` byte-identical in all four runs, as the setting's documentation states for
  every value.

Five is the most one pass can use: loaders beyond the first come from a process-wide pool of four
(`MAIN_LOADER_EXTRAS`), so concurrent chains share them. About 25 s on each of the 12 features
calls of an experiment. The default and the four documents that state it are changed.

**L12: rescore handoff, parked.**
- The handoff's encode (110-123 s per rescore) writes 65 GB at about 550 MB/s, while a buffered
  `dd` on the same disk writes 2.1 GB/s, so the disk does not bound it.
- A gdb profile shows about one busy thread. The batch transpose (`rows_into`) is parallel but
  works on 16,384-row batches.
- Larger batches (`MUMDIA_WIDE_SCAN=rowgroup` or `coalesced`) wrote the same bytes but measured
  slower (282 and 340 s). Those runs followed each other with 65 GB of dirty pages each, so the
  comparison is confounded and does not rule the lever out.
- The saving is at most about 2 min per rescore, 4 min per experiment. It needs a finer profile
  (main thread only, at a higher sampling rate) before any code. Parked under the rule to skip
  small wins that need much code.

### 7.6 L13: the second pass in the engine (2026-10-06)

`mbr.strategy: second_pass` with the `mbr.second_pass` sub-table (`lib_q` 0.05, `report_q` 0.01,
`min_rt_halfwidth_s` 0, `min_im_halfwidth` 0.04), `scripts/mbr_second_pass.py` (`build`, `report`)
and `run_experiment::second_pass`; docs/12 "mbr.strategy = second_pass", docs/24.
- It implements arm D only: predicted intensities, new decoys with both termini kept. Those are
  the settings section 4n validated.
- It is built for speed from the start:
  - the chains run under the `parallel_runs` plan and the memory budget (first chain alone,
    measured);
  - the library build skips the empirical-intensity pass, so no pass-1 quant runs before it;
  - inputs come from each run's retrace report, so nothing from pass 1 is read twice;
  - the quant traces are an extract and a retrace only (no features, no compete).

**Validation** (`quant_diag/m0/v_sp4.sh`). The engine run is the target pipeline with
`second_pass` at the 64 GB budget (`~/bin/mumdia-sp4`: 3f8516c plus the speed changes and L13). The
prototype (`m0_qd/sp_arms.sh`, arm D) ran on the same pass-1 IDs (`s1_dl45`).

| | prototype on the same IDs | engine |
|---|---|---|
| ions / global / eq / PB CV | 104,979 / 0.138 / 0.209 / 0.092 | 104,979 / 0.138 / 0.209 / 0.092 |
| DIA-NN-shared (eps / CV / E. coli log2) | 94,902: 0.132 / 0.094 / -1.81 | 94,902: 0.132 / 0.094 / -1.81 |
| MuMDIA-only | 10,077: 0.225 / 0.135 / -1.18 | 10,077: 0.225 / 0.135 / -1.18 |

- Pass-1 tables: 25 of 25 byte-identical to the L7 arm.
- Second-pass library (precursors, fragments, id map, all windows and seeds): 21 of 21
  byte-identical to the prototype's.
- Competed tables and the pooled second-pass rescore: the same content (the layout differs).
- Per run, the second pass accepts 100,073-101,896 rows at pooled q 0.01, against 91,199-92,998 in
  pass 1 (+8,600 to +9,690 per run).
- Against `eng_repick`'s arm D (section 4n: 105,078 / 0.137 / 0.209 / 0.092), this ID set is the
  4.5.0 end-to-end one, so the small differences are the ID set's.

**Cost of the second pass inside the engine run:** 7.5 min, against about 20 min for the
prototype and the 22 min of the transfer tiers it replaces.

| step | wall |
|---|---|
| library build | 57 s |
| six chains (ID chain plus quant traces; the first alone, then the budget plan) | 2.5 min |
| pooled rescore, 1,458,034 PSMs | 3.7 min |
| report filter and quant (two passes) | 20 s |

The run's total wall, 3:05:27, is not usable: another user's jobs held the host at load 40-100 and
the HDD at 70-96% busy during pass 1 (this run often had few cores of its own).

**Quiet-host rerun** (`v_sp4b.sh`, same binary and config):
- **1:49:37** wall, 2,800% CPU (28.0 of 64 cores on average);
- peak anonymous memory 42.7 GiB; the largest process (the rescore worker, through its mapped
  matrices) 130 GB;
- all 246 tables byte-identical to the contaminated run (the three files missing are ProteoBench
  outputs that only the first run's scoring wrote).

This is the whole target pipeline, pass 1 with the refit, the pooled rescore and the second pass,
at the 64 GB budget.

## 8. Where it stands (2026-10-06)

| | wall | peak (anonymous) |
|---|---|---|
| e2e_rx2 as measured before this roadmap (pass 1 + transfer MBR, shared host) | 3:29:14 | |
| M0: e2e_rx2 rerun, quiet host | 2:53:22 | 113.4 GiB |
| target pipeline under DeepLC 4.5.0, no second pass (`s1_dl45`) | 2:40:00 | |
| speed batch, no budget (`v_sp1`) | 1:46:20 | 89.2 GiB |
| speed batch, 64 GB budget (`v_sp1_b64`) | 1:51:54 | 32.6 GiB |
| **target pipeline with the second pass in the engine, 64 GB budget (`v_sp4b`)** | **1:49:37** | **42.7 GiB** |

- The batch reproduces the earlier tables byte for byte (level B), except L7 and L3, which passed
  the seed and entrapment gate (level S). L3 (`MUMDIA_NN_STORE=f16`) stays opt-in. A 64 GB node
  needs it for the pooled rescore, whose 60 GB matrix is file-backed and so does not show in the
  anonymous peak.
- Dropped after measurement: L6, L9, L14. Parked: L12 (rescore handoff, about 4 min).
- Not done: L3c (moments from the engine, one load pass under a memory limit), and the second
  pass's own levers (one extract for both trace sets; it is 7.5 min of the 110).
- Where the time goes now (`v_sp4b`): pass-1 chains 27.7 min, the refit's pass 2 about 35 min, the
  three rescores 25 min (workers 460 + 461 + 200 s), the multi-head calibrations 5.6 min, and
  convert and seed 3.9 min.

## 9. Why diaPASEF takes longer than Astral: the factor ladder (2026-10-06)

The six Astral HYE files take 25:39 to 36:06 end to end with MuMDIA on this host (robbin's runs,
128 threads, MBR off). One Astral run accepts 0.83M candidates and writes 12.4M chromatogram rows.
One diaPASEF run under the target pipeline accepts 6.94M and writes 104M (8.3x), and the pipeline
adds the raw-event retrace, the refit's second extraction and a second pooled rescore.

To separate these factors, the target pipeline's settings were added back one at a time to an
Astral-like configuration (`quant_diag/m0/ladder.sh`). All arms use the same binary (`mumdia-sp4`),
library and config base (`s1_dl45`), 64 threads, `mbr.strategy = none`, no memory budget and a fresh
cache per arm.

| arm | change | wall | peak RSS | accepted / run | ions | eq | PB CV | DIA-NN-shared |
|---|---|---|---|---|---|---|---|---|
| A0 | Astral-like: gate 0.2, no retrace, no repick, no IM features, no refit | 19:30 | 87 GiB | 2.3M | 58,649 | 0.194 | 0.080 | 56,555 |
| A1 | A0 + `gate_min_score` 0 | 38:17 | 121 GiB | 6.9M | 62,660 | 0.202 | 0.085 | 60,233 |
| A2 | A1 + retrace, repick, `retrace_apex`, `retain_top_peaks` 5, IM features | 1:11:40 (*) | 124 GiB | 6.9M | 86,623 | 0.180 | 0.084 | 81,524 |
| A3 | A2 + refit (`v_sp1`) | 1:46:20 | 89 GiB (anon.) | 6.9M | 94,151 | 0.180 | 0.084 | 87,918 |

(*) A2 shared the host from 20:34 with another job (about 38 cores, 389 GB). From the clean phases
of `v_sp4b` (pass-1 chains 32.8 min, pass-1 rescore 11.7 min), the clean A2 is estimated at about
65 min. Ions are from rescore seed 0. A single seed moves ions by about 1%, and every step here is
far larger than that.

Where each step spends its time (seconds, from `m0_stages.py`):

| phase | A0 | A1 | A2 |
|---|---|---|---|
| convert, seed, multi-head r0 | 454 | 409 | 418 |
| pass-1 r0 alone (extract, retrace, features) | 89 | 176 | 304 |
| pass-1 chains r1-r5, concurrent | 360 | 744 | 1,971 (*) |
| pooled rescore, worker | 226 | 520 | 828 (*) |
| rescore q, quant, LFQ, report | 40 | 446 | 778 |

Findings:

- **A0 is Astral speed.** At Astral-like settings the six diaPASEF files take 19.5 min, the same
  range as the Astral runs. The extra time is not a diaPASEF penalty in the engine. It comes from
  the settings that make diaPASEF sensitive.
- **Gate 0 (A1) costs 1.96x wall for +6.8% ions.** It triples the candidates (2.3M to 6.9M per run)
  and the chromatogram rows (35M to 104M). That cost goes into extract and features, into the rescore
  worker (2.3x), and into quant. Pass-1 quant reads each run's 17 GB chromatogram table from the
  HDD to keep 1M of its 104M rows: 435-501 s with six runs at once, against 38 s in A0. In the
  target pipeline the second pass replaces this read (`v_sp4b`: quant 0.3 min), so it is not a
  lever there. It is a lever for any configuration without the second pass.
- **The retrace block (A2) costs the most time and gives the most ions: +38% ions, +81% DIA-NN-shared
  precursors over A1**, and eq improves from 0.202 to 0.180. Most of its cost is retrace inside the
  chains (157 s for r0 alone, and memory-bandwidth bound when concurrent, section 3.4).
- **The refit (A3) costs 35 to 41 min (measured A2 or clean estimate) for +8.7% ions** (+7.8% shared), at the same eq and CV.
- Per ion found, the steps cost about: A0 to A1, 1.0 min per 1,000 ions; A1 to A2, about 1.1 min
  (clean estimate); A2 to A3, 4.6 to 5.5 min. The refit is the most expensive step per ion. Every step
  stays above the 1% rule of this roadmap.

The 800 s quoted for six HYE files was not reproduced here. The closest arm is A0 (19.5 min, of
which 3.6 min is the multi-head calibration). More threads or a warm projection cache could bring an
A0-like setting near 800 s, but that was not measured. The figure does not apply to the target
pipeline, which does about 8x the extraction work per run and two rescores.
