"""PyTorch semi-supervised NN rescorer sidecar (Stage F, RescorerKind::NnTorch).

Usage:
    python nn_rescore_worker.py <input.pin> <output.parquet>

Reads a Percolator PIN, rescores it with a PyTorch MLP trained in the
Percolator/mokapot semi-supervised scheme, writes `candidate_id` (the SpecId tail)
+ `score` + `q_value`. Scores EVERY PSM (targets and decoys) so target-decoy FDR
downstream is intact. Same positional-CLI file contract as `mokapot_worker.py`;
select it with `rescore.classifier = "nn_torch"` and point `rescore.python` at an
interpreter with torch + pandas + pyarrow.

Algorithm (per CV fold, so every PSM is scored out-of-fold by a model that never
trained on it): initialise from the best single feature+sign, then iterate
{recompute target-decoy q on the training folds -> targets at q<=train_fdr are
positives, all decoys negatives -> train the MLP from scratch -> rescore} for
`iters` rounds; score the held-out fold with the final model.

MEMORY (multi-run / large PINs): two feature backends behind one accessor.
    - in-memory (default for PINs <= MUMDIA_NN_STREAM_GB, 4 GB): the full standardised
      feature matrix is held in RAM, and not much else: one n x features float32 block plus
      one fold's gathered training rows. The load is streamed row group by row group and
      its transients are returned to the OS before training starts. Mean/std standardisation, the same as the streaming
      backend: one transform for every backend and every handoff, so a score does not
      depend on which one ran (docs/31 F5).
    - streaming memmap (large PINs, or MUMDIA_NN_STREAM=1): the PIN is read ONCE in
      chunks into a disk-backed float32 memmap (the same mean/std, accumulated in the
      same pass); training and scoring then draw MINIBATCHES indexed into the
      memmap, so peak RAM is one batch + per-row metadata, NOT the whole matrix.
      This is what makes combining many runs into one rescoring tractable: the full
      PIN never lives in RAM at once.

Determinism note (docs/14_build_test_deploy_gotchas.md): NN training is only
approximately reproducible. Set MUMDIA_NN_SEEDS>1 to ensemble seeds and average
out-of-fold scores.

Input format: either the legacy tab-separated PIN or a Parquet feature table, chosen by the
file extension (.parquet / .pq). Parquet avoids serialising the whole feature matrix as text -
measured at 34% of a rescore on a 1.5M-row subset, and worse at full scale where a 30 GB text
PIN also forced the streaming backend. Column names and semantics are identical either way.

Env knobs (all optional):
    MUMDIA_NN_FOLDS       = 3        cross-validation folds
    MUMDIA_NN_ITERS       = 5        semi-supervised self-training iterations
    MUMDIA_NN_EPOCHS      = 25       NN epochs per iteration
    MUMDIA_NN_HIDDEN      = "128,64" comma-separated hidden layer sizes
    MUMDIA_NN_DROPOUT     = 0.3
    MUMDIA_NN_LR          = 1e-3
    MUMDIA_NN_WD          = 1e-4     weight decay
    MUMDIA_NN_BATCH       = 4096
    MUMDIA_NN_TRAIN_FDR   = 0.01     positive-selection FDR during training
    MUMDIA_NN_SEEDS       = 1        seed models to ensemble (average OOF)
    MUMDIA_NN_SEED        = 0        base seed; ensemble member s uses SEED + s (seeded repeats)
    MUMDIA_NN_STREAM      = auto     auto|1|0  force the streaming memmap backend
    MUMDIA_NN_STREAM_GB   = auto     auto-stream when the decoded feature matrix exceeds
                                     this many GB. Unset, it is TWICE free physical
                                     memory, never below the historical 4 GB: the memmap
                                     measured ~9x slower than RAM (166 min against ~20 on
                                     a 4.52 GB matrix), so it is a last resort and the
                                     page file absorbs a matrix that merely overflows.
                                     Set it explicitly on a machine with no page file.
    MUMDIA_NN_CHUNK       = 250000   PIN rows per read chunk (streaming backend)
    MUMDIA_NN_INIT_SAMPLE = 300000   rows used to pick the init feature. When no feature
                                     reaches the training FDR on that sample the scan is
                                     repeated on 4x the rows, up to the whole fold: on an
                                     8.07M-row immunopeptidomics pool 300k rows (3.7%)
                                     held too few true PSMs for any of 347 features to
                                     pass 1%, the scan returned an arbitrary feature with
                                     0 and the fold aborted, while the same features gave
                                     150-217 at 1% on a 2M pool of the same run.
    MUMDIA_NN_INIT_FDR_MAX = 0.05    first-iteration bootstrap only: when the init feature
                                     selects no positive at the training FDR over the
                                     whole fold, the threshold is loosened in steps
                                     (0.02, 0.05, 0.1) up to this ceiling for that one
                                     selection; every later iteration re-selects at the
                                     training FDR on the model's own scores. 0 disables
                                     the ladder and restores the hard error.
    MUMDIA_NN_INIT_TOPK   = 0        > 0 sorts only a top-k window per feature in the init
                                     scan instead of the whole sample (7.2x on that phase).
                                     NOT exact: tie ordering at the window edge shifted 30 of
                                     774 counts by 1-2 in testing, though the chosen feature
                                     was unchanged. 0 (default) keeps the exact full sort.
    MUMDIA_NN_EARLY_STOP  = 1        stop self-training once the positive set stabilises
    MUMDIA_NN_EARLY_STOP_TOL = 0.01  churn tolerance for that stop; 0 means exact equality,
                                     which measurably never triggers (dropout + retraining
                                     flips a few borderline PSMs every iteration forever).
                                     Measured on a 40k pool at iters=10: tol 0.01 -> 1.59x
                                     for -0.4% peptides (within NN noise); 0.03 -> 2.15x
                                     for -1.0%; 0.002 -> only 1.06x.
    MUMDIA_NN_WARM_START  = 0        1 reuses the previous iteration's weights AND Adam
                                     state instead of rebuilding the model from random
                                     initialisation every iteration. The from-scratch default
                                     is Percolator's behaviour and is why EPOCHS=25 is needed
                                     each time; warm starting lets later iterations use far
                                     fewer (see MUMDIA_NN_WARM_EPOCHS). Changes the training
                                     trajectory, so gate on peptides + decoy% before enabling.
    MUMDIA_NN_WARM_EPOCHS = 0        epochs per iteration once warm-started (0 = keep EPOCHS).
                                     Applies from the second iteration on; the first still
                                     runs the full EPOCHS to leave random initialisation.
    MUMDIA_NN_TRAIN_SUB   = 0        subsample the per-iteration TRAINING rows: a fraction
                                     in (0, 1], or an absolute row cap if > 1. 0 (default)
                                     trains on every selected positive and every decoy, the
                                     historical behaviour. Stratified, so class balance and
                                     pos_weight are preserved; positive SELECTION still runs
                                     over the full fold, so this trades gradient steps for
                                     wall time without narrowing what can be discovered.
    MUMDIA_NN_NEG_SELECT  = random   which decoys survive NEG_RATIO: random | margin |
                                     hybrid. `margin` keeps the highest-scoring (hardest)
                                     decoys under the current model, `hybrid` splits the
                                     budget half hard / half random. Only meaningful with
                                     NEG_RATIO > 0.
    MUMDIA_NN_NEG_RATIO   = 0        cap TRAINING negatives at this multiple of the
                                     positives selected in the same iteration (e.g. 3 = at
                                     most 3 decoys per positive). 0 (default) trains on every
                                     decoy in the fold, which is ~15:1 in practice, so most
                                     gradient steps go to negatives. FDR is unaffected -
                                     decoys are thinned for TRAINING only; scoring,
                                     target/decoy competition and q-values still use the full
                                     pool. pos_weight is recomputed from the capped set.
    MUMDIA_NN_FEATURES    = ""       restrict rescoring to these features: comma-separated
                                     names, or a path to a file with one name per line.
                                     Applied before the PIN is read, so dropped columns are
                                     never parsed or moved. Empty (default) uses all.
    MUMDIA_NN_DEVICE      = auto     auto|cuda|cpu. auto uses the GPU when torch can see
                                     one. Forcing gives a device-only comparison within one
                                     environment; cuda errors out rather than silently
                                     falling back to CPU on a CPU-only torch build.
    MUMDIA_NN_THREADS     = 16       torch CPU threads asked for (0 = leave torch's default);
                                     the engine passes its --threads here
    MUMDIA_NN_DROP_CONSTANT = 1      drop feature columns that are constant over the pool
                                     (from the parquet footer's per-column statistics, so no
                                     read). A constant column standardises to exactly 0 and
                                     contributes nothing to any prediction, but its first-layer
                                     weights see only Adam's L2 term and decay below 1.2e-38;
                                     on Intel cores every FMA on a subnormal operand then takes
                                     a ~100x microcode assist, which made a desktop rescore 6x
                                     slower than the same pool on an EPYC. 0 = keep them.
    MUMDIA_NN_CLAMP_TINY  = 1e-20    once per epoch, set every parameter and buffer with
                                     |value| below this to exactly 0. Removes the subnormal
                                     source itself: Adam + L2 shrinks parameters that get no
                                     data gradient (dead units, their BatchNorm scale/shift and
                                     running variances) geometrically until they cross 1.2e-38;
                                     a value below 1e-20 contributes nothing to a float32 sum, a
                                     weight of exactly 0 stays 0, and 0 x anything is 0. Works on
                                     every CPU without touching the FPU. 0 = off.
    MUMDIA_NN_FLUSH_DENORMAL = 1     flush subnormal float32 to zero in torch (FTZ/DAZ). The
                                     constant columns are not the only subnormal source: dead
                                     hidden units and their BatchNorm buffers decay too (census
                                     on the six-run pool: up to ~6,400 parameters, 11 buffers,
                                     ~3,000 activations per round), and without this the desktop
                                     rescore took 61.8 min against 11.4 with it. Any change to
                                     the arithmetic reshuffles a single seed's count by up to
                                     ~0.4% (thread count, column set, this flag alike); means
                                     over seeds are flat. 0 = leave the FPU default.
    MUMDIA_NN_DEBUG_DENORMALS = 0    1 = after every training round, count subnormal cells in
                                     the model's parameters, buffers and one chunk's
                                     activations, and print them (diagnostic only)
    MUMDIA_NN_THREAD_CAP  = auto     ceiling on those threads. auto = the performance-core
                                     count on a hybrid (P+E core) CPU, else 16. 0 = no cap.
                                     Measured: the MLP is flat from 16 to 64 threads on two
                                     EPYC generations, and on an i9-13900KS (8P+16E) 30
                                     threads took 6x longer than 8 on a 3.1M-row pool.
    MUMDIA_NN_PREGATHER_GB= 8        pre-gather the fold's training rows when they fit in
                                     this many GB (one gather per iteration instead of a
                                     fancy-index copy per minibatch)
    MUMDIA_NN_LOAD_THREADS = auto    threads that fill the in-memory matrix from a parquet
                                     handoff: each 32,768-row moment sub-block of a row group
                                     is written straight into its rows and its float64 sums
                                     are added in the original order, so the matrix, mean and
                                     std are byte-identical. auto = min(8, torch CPU threads);
                                     each thread holds a 32,768 x features float64 buffer.
                                     0 = the original serial loop (no read-ahead/pre_buffer).
    MUMDIA_NN_READ_AHEAD  = 1        decode row group r+1 on a reader thread while r is being
                                     filled (at most two decoded groups resident). 0 = off.
    MUMDIA_NN_PRE_BUFFER  = 1        open the reader's ParquetFile with pre_buffer=True
                                     (coalesced column-chunk reads). 0 = off.
    MUMDIA_NN_SCAN_THREADS = auto    threads for the init feature scan (one column, both
                                     signs, per task; the winner is reduced in the serial
                                     (column, sign) order, so the choice is identical).
                                     auto = the torch CPU thread count, capped by the sample
                                     size (one thread per 20,000 rows). 1 = serial.
    MUMDIA_NN_SELECT      = window   how each round selects its positives: window = sort
                                     only the top rows down to the (floor(fdr * targets)+1)-th
                                     best decoy, certify that no later row can be accepted,
                                     else fall back to the full sort; the hybrid/margin decoy
                                     order uses one uint64 key sort. Both reproduce the stable
                                     argsort exactly. full = the previous full sort each round.
    MUMDIA_NN_GATHER      = torch    how `score_idx` gathers a scoring batch on the
                                     in-memory backend: torch = `torch.index_select` into
                                     one reused buffer (multi-threaded); numpy = the
                                     previous `Xs[idx]` fancy index. Same values, same
                                     shapes, byte-identical scores. The streaming backend
                                     always uses the numpy path.
    MUMDIA_NN_PARALLEL    = 0        opt-in: > 0 trains the (seed, fold) tasks in that many
                                     spawned processes at once. The matrix is shared through
                                     a read-only memmap next to the output (the in-memory
                                     backend writes it there instead of to RAM), not copied
                                     per process. The epoch shuffle is then keyed per (seed,
                                     fold, iteration, epoch) rather than drawn from one
                                     stream per seed, which changes the scores once, like a
                                     seed change; they do not depend on the process count
                                     (1 runs the same keyed tasks one after another in one
                                     child). 0 = the serial loop and today's scores.
    MUMDIA_NN_PARALLEL_THREADS = auto  torch threads per process under MUMDIA_NN_PARALLEL;
                                     auto = the serial worker's resolved thread count. Fixed
                                     per process, so the arithmetic of a task does not depend
                                     on how many processes run.
    MUMDIA_NN_FINAL_POOL_SCORE = 0   1 = also score the training pool after the LAST
                                     round and log its target count. Those scores feed no
                                     later selection, so by default the pass is skipped and
                                     the per-fold log line reports the held-out fold's
                                     count instead. Scores are identical either way.

Performance notes (measured on a 32-core CPU box, 1.3M-PSM rescore):
    The MLP is tiny (387->128->64->1, ~58k params) but is executed ~160,500 times
    (batches x epochs x iters x folds), so wall time is dominated by STEP COUNT, not by
    model size: cutting the model to 15k params changed runtime by 1.02x, while
    `ITERS` (linear) and `BATCH` (1.5x from 4096->16384, per-row cost 1.60->1.04 us) do
    move it. Threads saturate at 16 (32 is slower than 16). The per-minibatch
    `Xs[idx]` fancy-index gather was ~25% of runtime; `PREGATHER_GB` removes it.
"""

import hashlib
import os
import re
import sys
import threading
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

NON_FEATURE = {"SpecId", "Label", "ScanNr", "ExpMass", "CalcMass", "Peptide", "Proteins"}


def env_f(name, default):
    return float(os.environ.get(name, default))


def env_i(name, default):
    return int(os.environ.get(name, default))


def strip_pep(p):
    s = re.sub(r"\[[^\]]*\]", "", str(p))
    s = re.sub(r"^[A-Z-]\.", "", s)
    s = re.sub(r"\.[A-Z-]$", "", s)
    return s


def peptide_fold(peptide, folds):
    """Fallback fold index, hashed from the mod-stripped peptidoform.

    Does NOT place a target and its paired decoy in the same fold: `strip_pep` leaves
    the `DECOY_` marker in place, and for a reverse-decoy library the decoy peptidoform
    is the reversed sequence, so no string derived from it reaches its target. Kept only
    for a PIN written without fold keys.
    """
    return int(hashlib.md5(strip_pep(peptide).encode()).hexdigest(), 16) % folds


def folds_for(peptides, fold_keys, folds, off=0):
    """Fold index per row, preferring the engine's explicit keys.

    `fold_keys` is `base_peptide_id` per PIN row, which a target and its paired decoy
    share on every library path, so keying on it puts them in the same fold. That is
    what `percolator_lite` does and what docs/11 claims this worker does; the hashed
    fallback did not. `off` is the flat row offset, which is what makes one helper
    serve the chunked streaming backend as well as the two whole-table ones.
    """
    if fold_keys is not None:
        want = len(peptides)
        got = fold_keys[off:off + want]
        # Numpy slicing past the end returns a SHORT array rather than raising, and a
        # short `fold` puts the tail rows in no fold at all: `np.where(fold == f)` cannot
        # index them, they are never scored, and they keep the zero initialiser, which the
        # final rank-normalisation turns into a plausible tied mid-rank score. The Rust
        # caller's completeness contract is satisfied by that, so `rescore.strict` does not
        # catch it either (docs/31 F5).
        if len(got) != want:
            raise SystemExit(
                "MUMDIA_NN_FOLD_KEYS has %d rows but the PIN needs at least %d "
                "(rows %d..%d); the companion table does not belong to this PIN, and "
                "folding on a truncated one would leave the tail unscored."
                % (len(fold_keys), off + want, off, off + want))
        return (got % folds).astype(np.int16)
    return np.array([peptide_fold(x, folds) for x in peptides], np.int16)


# Beyond this many torch threads the rescorer's MLP does not get faster on any CPU measured
# (EPYC 7H12: 16 = 32 threads, 64 slower; EPYC 9354: 8 = 32), so more only costs the rest of
# the machine. It is a ceiling on what the engine's --threads asks for, not a target.
_THREAD_CAP = 16


def performance_cores():
    """Performance-core count on a hybrid CPU, or None when the CPU is uniform or unknown.

    Windows only: `GetLogicalProcessorInformationEx(RelationProcessorCore)` reports an
    EfficiencyClass per physical core, and the highest class is the performance tier. On
    an i9-13900KS that is 8 (with 16 efficiency cores). Every OpenMP-parallel op waits for
    its slowest thread, so a thread pinned to an efficiency core, or sharing a performance
    core's second hyperthread, sets the pace for all of them: measured on that CPU, one
    rescore of a 522k-row pool took 152 s at 4 threads and 200 s at 30, and the same ratio
    grew with pool size (docs/28). Linux desktops with hybrid CPUs are not detected here;
    MUMDIA_NN_THREAD_CAP sets the ceiling explicitly there.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import struct
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        relation_processor_core = 0
        size = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(
            relation_processor_core, buf, ctypes.byref(size)
        ):
            return None
        raw = buf.raw
        classes = []
        off = 0
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship (u32), Size (u32), then a
        # PROCESSOR_RELATIONSHIP whose second byte is EfficiencyClass. One record per core.
        while off + 10 <= size.value:
            relationship, record_size = struct.unpack_from("<II", raw, off)
            if record_size < 10:
                return None
            if relationship == relation_processor_core:
                classes.append(raw[off + 9])
            off += record_size
        if len(set(classes)) < 2:
            return None
        top = max(classes)
        return sum(1 for c in classes if c == top)
    except Exception:  # noqa: BLE001 - detection is best effort; None keeps the flat cap
        return None


def torch_thread_cap():
    """`(cap, why)` for the torch CPU thread count; 0 means uncapped."""
    raw = os.environ.get("MUMDIA_NN_THREAD_CAP", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            return max(0, int(float(raw))), "MUMDIA_NN_THREAD_CAP"
        except ValueError:
            pass
    p_cores = performance_cores()
    if p_cores:
        return p_cores, f"{p_cores} performance cores on a hybrid CPU"
    return _THREAD_CAP, "flat beyond this on every CPU measured"


def constant_columns(pin_path, cols):
    """Feature columns whose parquet footer statistics show a single value in every row group.

    No data is read: the writer records min/max per column chunk, and a column with
    min == max across all groups is constant over the pool. Columns without statistics are
    kept, so a file from a writer that omits them behaves as before.
    """
    pf = pq.ParquetFile(pin_path)
    md = pf.metadata
    position = {name: j for j, name in enumerate(pf.schema_arrow.names)}
    out = []
    for c in cols:
        j = position.get(c)
        if j is None:
            continue
        lo = hi = None
        complete = True
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(j).statistics
            if st is None or not st.has_min_max:
                complete = False
                break
            lo = st.min if lo is None else min(lo, st.min)
            hi = st.max if hi is None else max(hi, st.max)
        if complete and lo is not None and lo == hi:
            out.append(c)
    return out


def _optimizer_tensors(opt):
    """Adam's moment tensors: they decay geometrically for a zero-gradient parameter too."""
    if opt is None:
        return []
    out = []
    for state in opt.state.values():
        for v in state.values():
            if hasattr(v, "is_floating_point") and v.is_floating_point() and v.dim() > 0:
                out.append(v)
    return out


def _clamp_tiny(model, threshold, opt=None):
    """Set every parameter, floating buffer and optimizer moment with |value| < threshold to 0.

    Adam with L2 decay shrinks a parameter that receives no data gradient by a roughly
    constant factor per step (the L2 term is normalised by a second moment that decays much
    more slowly), so dead units' weights, their BatchNorm scale and shift and the running
    variances head for the subnormal range within a few thousand steps. Adam's own moment
    tensors follow: for a zero-gradient parameter the first moment decays as 0.9^k and the
    second as 0.999^k, and every optimizer step then touches them elementwise. Below 1e-20 a value
    adds nothing to any float32 sum that matters here, a weight of exactly 0 stays 0 under
    Adam, and 0 times anything is 0, so after this no subnormal can arise in a weight, a
    buffer or an activation. Once per epoch over ~60k values: free.
    """
    if threshold <= 0:
        return
    import torch  # imported here: the module imports torch inside main(), after the device check

    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()) + _optimizer_tensors(opt):
            if t.is_floating_point():
                t.masked_fill_(t.abs() < threshold, 0.0)


def _denormal_census(model, xb, opt=None):
    """Subnormal float32 cells in the model's parameters, buffers and one chunk's activations.

    Diagnostic for the flush-to-zero switch above: subnormals in any of the three are what
    make every Linear on an Intel core pay the microcode assist.
    """
    import torch  # imported here: the module imports torch inside main(), after the device check

    tiny = 1.1754944e-38

    def count(t):
        a = t.detach().abs().float()
        return int(((a > 0) & (a < tiny)).sum())

    params = sum(count(p) for p in model.parameters())
    bufs = sum(count(b) for b in model.buffers())
    acts = 0
    with torch.no_grad():
        h = xb
        for layer in model.net:
            h = layer(h)
            acts += count(h)
    moments = sum(count(t) for t in _optimizer_tensors(opt))
    return params, bufs, acts, moments


def _accumulate_moments(blk, s1, s2, rows=32768):
    """Add a float32 block's per-column sum and sum of squares to `s1`/`s2`, in float64.

    In sub-blocks: `(blk.astype(np.float64) ** 2)` on a 250k-row chunk is two 0.77 GB
    temporaries per chunk, and that transient -- not the matrix -- set the worker's
    high-water mark during the load.
    """
    for a in range(0, len(blk), rows):
        sub = blk[a:a + rows].astype(np.float64)
        s1 += sub.sum(axis=0)
        np.square(sub, out=sub)
        s2 += sub.sum(axis=0)


# Rows per moment sub-block. The per-column float64 sums are accumulated one sub-block at a
# time, in this partition, so any loader that reproduces the partition reproduces the sums.
_MOMENT_ROWS = 32768


def moment_blocks(rows, chunk, sub=_MOMENT_ROWS):
    """(start, stop) rows of one row group's moment sub-blocks, in accumulation order.

    The partition of the original loop: the row group is sliced into `chunk`-row pieces from
    its first row, and each piece into `sub`-row sub-blocks from the piece's first row. The
    float64 sums depend on this partition (addition is not associative), so it is kept.
    """
    out = []
    for c0 in range(0, rows, chunk):
        k = min(chunk, rows - c0)
        for a in range(0, k, sub):
            out.append((c0 + a, c0 + min(a + sub, k)))
    return out


def moments_to_mean_std(s1, s2, n):
    """Mean and standard deviation (float32) from float64 column sums, as every backend does."""
    mean = (s1 / n).astype(np.float32)
    std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def _column_to_numpy(col):
    """A decoded column as numpy, nulls as NaN (older pyarrow lacks the keyword)."""
    try:
        return col.to_numpy(zero_copy_only=False)
    except TypeError:
        return col.to_numpy()


def _open_parquet(path, pre_buffer):
    """`pq.ParquetFile`, with `pre_buffer` (coalesced column-chunk reads) when available."""
    if pre_buffer:
        try:
            return pq.ParquetFile(path, pre_buffer=True)
        except TypeError:  # pragma: no cover - pyarrow without the keyword
            pass
    return pq.ParquetFile(path)


_TLS = threading.local()


def _fill_block(arrays, dst, r0, r1):
    """Write rows `r0:r1` of one decoded row group into `dst` and return their moments.

    `dst` is the matrix's view of exactly those rows. Each column is narrowed to float32
    first (the cast the original `blk[:, j] = column` assignment made), then its non-finite
    cells are set to 0.0 on that narrowed column, which is what `np.nan_to_num` did to the
    block: a finite float64 beyond float32 range becomes inf in the cast and 0.0 here, as
    before. The moments are the original sub-block's: the rows cast to a C-contiguous
    float64 block of the same shape, summed, squared in place and summed again. The float64
    block is a per-thread buffer, reused across calls.
    """
    for j, a in enumerate(arrays):
        src = a[r0:r1]
        if src.dtype != np.float32:
            src = src.astype(np.float32)
        finite = np.isfinite(src)
        if not finite.all():
            src = np.where(finite, src, np.float32(0.0))
        dst[:, j] = src
    k = r1 - r0
    nf = dst.shape[1]
    buf = getattr(_TLS, "moments", None)
    if buf is None or buf.shape[1] != nf or buf.shape[0] < k:
        buf = _TLS.moments = np.empty((max(k, _MOMENT_ROWS), nf), np.float64)
    sub = buf[:k]
    np.copyto(sub, dst)
    p1 = sub.sum(axis=0)
    np.square(sub, out=sub)
    p2 = sub.sum(axis=0)
    return p1, p2


def fill_parquet_matrix(pin_path, feat_cols, n, chunk, out, threads=1, read_ahead=True,
                        pre_buffer=True, initializer=None):
    """Decode the handoff's feature columns into `out` (n x nf float32); return (s1, s2).

    Byte-for-byte the matrix and the float64 column sums of the original loop
    (`_fill_parquet_matrix_legacy`), produced faster:
      - `read_ahead`: one reader thread, with its own ParquetFile, decodes row group r+1
        while row group r is filled, so at most two decoded groups are held;
      - `pre_buffer`: that ParquetFile coalesces column-chunk reads;
      - `threads`: the moment sub-blocks of a group (see `moment_blocks`) are filled into
        disjoint row ranges of `out` in parallel, straight from the decoded columns, with no
        intermediate block, and their partial sums are added in sub-block order.
    `initializer` runs once per pool thread (the worker's flush-to-zero setting).
    """
    from concurrent.futures import ThreadPoolExecutor

    nf = len(feat_cols)
    s1 = np.zeros(nf, np.float64)
    s2 = np.zeros(nf, np.float64)
    nrg = pq.read_metadata(pin_path).num_row_groups
    state = {}

    def read(rg):
        t0 = time.time()
        pf = state.get("pf")
        if pf is None:
            pf = state["pf"] = _open_parquet(pin_path, pre_buffer)
        tbl = pf.read_row_group(rg, columns=feat_cols)
        rows = tbl.num_rows
        arrays = [_column_to_numpy(tbl.column(j)) for j in range(tbl.num_columns)]
        del tbl
        return rows, arrays, time.time() - t0

    reader = ThreadPoolExecutor(max_workers=1, initializer=initializer) if read_ahead else None
    pool = (ThreadPoolExecutor(max_workers=int(threads), initializer=initializer)
            if threads > 1 else None)
    off = 0
    try:
        pending = reader.submit(read, 0) if (reader is not None and nrg) else None
        for rg in range(nrg):
            tw = time.time()
            if reader is not None:
                rows, arrays, busy = pending.result()
                _detail("load: read wait", time.time() - tw)
                pending = reader.submit(read, rg + 1) if rg + 1 < nrg else None
            else:
                rows, arrays, busy = read(rg)
            _detail("load: read", busy)
            tf = time.time()
            if off + rows > n:
                raise RuntimeError(
                    f"parquet row mismatch: metadata {n}, features at least {off + rows}")
            blocks = moment_blocks(rows, chunk)
            if pool is not None:
                futs = [pool.submit(_fill_block, arrays, out[off + a:off + b], a, b)
                        for a, b in blocks]
                parts = [fu.result() for fu in futs]
            else:
                parts = [_fill_block(arrays, out[off + a:off + b], a, b) for a, b in blocks]
            for p1, p2 in parts:
                s1 += p1
                s2 += p2
            off += rows
            del arrays, parts
            _detail("load: fill + moments", time.time() - tf)
    finally:
        if reader is not None:
            reader.shutdown(wait=True)
        if pool is not None:
            pool.shutdown(wait=True)
        state.clear()
    if off != n:
        raise RuntimeError(f"parquet row mismatch: metadata {n}, features {off}")
    return s1, s2


def _fill_parquet_matrix_legacy(pin_path, feat_cols, n, chunk, out):
    """The original serial load loop (MUMDIA_NN_LOAD_THREADS=0); returns (s1, s2)."""
    nf = len(feat_cols)
    s1 = np.zeros(nf, np.float64)
    s2 = np.zeros(nf, np.float64)
    _pf = pq.ParquetFile(pin_path)
    off = 0
    _tbl = _b = None
    # One row group at a time. `iter_batches` reads ahead and decodes groups in
    # parallel, and its buffered batches grew into a second copy of the matrix:
    # measured, the worker climbed to 11.2 GB while filling a 4.85 GB `Xs`, then
    # fell to 6.3 GB the moment the loop ended. Reading a group, slicing it into
    # CHUNK-row blocks and dropping it bounds the transient to one group, which the
    # engine writes at 131,072 rows (200 MB at 387 features); a file with parquet's
    # default 1,048,576-row groups still loads, at 1.6 GB per group.
    for _rg in range(_pf.num_row_groups):
        _tr = time.time()
        _tbl = _pf.read_row_group(_rg, columns=feat_cols)
        _tf = time.time()
        _detail("load: read", _tf - _tr)
        for _s0 in range(0, _tbl.num_rows, chunk):
            _b = _tbl.slice(_s0, chunk)
            k = _b.num_rows
            blk = np.empty((k, nf), np.float32)
            for j in range(nf):
                blk[:, j] = _b.column(j).to_numpy(zero_copy_only=False)
            np.nan_to_num(blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            out[off:off + k] = blk
            _accumulate_moments(blk, s1, s2)
            off += k
            del blk
        del _tbl, _b
        _tbl = _b = None
        _detail("load: fill + moments", time.time() - _tf)
    del _pf, _tbl, _b
    if off != n:
        raise RuntimeError(f"parquet row mismatch: metadata {n}, features {off}")
    return s1, s2


def _standardise_rows(X, mean, std, i0, i1):
    view = X[i0:i1]
    np.subtract(view, mean, out=view)
    np.divide(view, std, out=view)
    np.clip(view, -8, 8, out=view)


def standardise_matrix(X, mean, std, chunk, threads=1, initializer=None, block=65536):
    """`X = clip((X - mean) / std, -8, 8)` in place.

    Every operation is elementwise, so the bytes do not depend on how the rows are
    partitioned; with `threads > 1` row blocks are processed on a thread pool. `threads <= 1`
    is the original serial loop over `chunk`-row views.
    """
    n = X.shape[0]
    if threads <= 1:
        # In place on each chunk: `(Xs[i:j] - mean) / std` made two chunk-sized copies.
        for i in range(0, n, chunk):
            _standardise_rows(X, mean, std, i, i + chunk)
        return
    from concurrent.futures import ThreadPoolExecutor

    step = max(1, min(int(chunk), int(block)))
    with ThreadPoolExecutor(max_workers=int(threads), initializer=initializer) as ex:
        list(ex.map(lambda i: _standardise_rows(X, mean, std, i, i + step),
                    range(0, n, step)))


def _row_ids(spec_ids):
    """`<prefix>_<flat row>` -> int64 flat rows, without a Python string per row."""
    tail = pc.replace_substring_regex(spec_ids, pattern="^.*_", replacement="")
    return pc.cast(tail, pa.int64()).to_numpy()


def _release_allocator_slack():
    """Return freed memory to the OS after a bulk load.

    Arrow's pool and glibc both keep freed pages mapped by default, so RSS after the load
    stays at the load's transient peak rather than at what is still referenced. Every call
    here is best effort and platform-dependent.
    """
    import gc

    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001 - older pyarrow without the call
        pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # noqa: BLE001 - not glibc
            pass


PHASE = {}
# How the self-training rounds selected their positives: from a certified window, or with
# the full sort. Printed with the sub-timers.
SELECT_COUNTS = {}

# Sub-timers, kept apart from PHASE so `sum(PHASE.values())` stays a sum of disjoint wall
# intervals. The load entries split `1_pin_read_standardise` (so they are already inside
# that phase); `selection` is the positive re-selection between a pool score and the next
# training round, which no phase covers.
DETAIL = {}


def _tick(name, t0):
    "Accumulate elapsed wall time under `name` and return a fresh timestamp."
    PHASE[name] = PHASE.get(name, 0.0) + (time.time() - t0)
    return time.time()


def _detail(name, seconds):
    "Accumulate `seconds` under the sub-timer `name` (never into PHASE)."
    DETAIL[name] = DETAIL.get(name, 0.0) + seconds


def _print_timers():
    """The phase breakdown, then the sub-timers that are not part of its total."""
    tot = sum(PHASE.values())
    print("nn_rescore_worker: phase breakdown (wall seconds)", flush=True)
    for k in sorted(PHASE):
        v = PHASE[k]
        print("    %-26s %8.1f s  %5.1f%%" % (k[2:], v, 100 * v / max(tot, 1e-9)), flush=True)
    print("    %-26s %8.1f s" % ("MEASURED TOTAL", tot), flush=True)
    if DETAIL:
        print("nn_rescore_worker: sub-timers (wall seconds; not added to the total above)",
              flush=True)
        for k in sorted(DETAIL):
            print("    %-26s %8.1f s" % (k, DETAIL[k]), flush=True)
    if CHILD_PHASE:
        print("nn_rescore_worker: parallel tasks, summed over all tasks (process seconds; "
              "they overlap in wall time)", flush=True)
        for k in sorted(CHILD_PHASE):
            print("    %-26s %8.1f s" % (k[2:], CHILD_PHASE[k]), flush=True)
        for k in sorted(CHILD_DETAIL):
            print("    %-26s %8.1f s" % (k, CHILD_DETAIL[k]), flush=True)
    if SELECT_COUNTS:
        print("nn_rescore_worker: positive selection: %d from a certified window, %d full "
              "sort(s)" % (SELECT_COUNTS.get("window", 0), SELECT_COUNTS.get("full", 0)),
              flush=True)


def tda_q(scores, is_target):
    """Target-decoy q-values. scores desc; FDR=(decoys+1)/max(1,targets), q=running min."""
    order = np.argsort(-scores, kind="stable")
    t = is_target[order].astype(float)
    ct = np.cumsum(t)
    cd = np.cumsum(1 - t)
    fdr = (cd + 1) / np.maximum(ct, 1)
    q = np.minimum.accumulate(fdr[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out


def n_targets_at(scores, is_target, fdr):
    q = tda_q(scores, is_target)
    return int(((q <= fdr) & (is_target == 1)).sum())


def desc_order(scores):
    """`np.argsort(-scores, kind="stable")` for a float32 vector, from one uint64 sort.

    Each score is mapped to an order-preserving 32-bit key in the high half of a uint64 and
    its position fills the low half, so the keys are distinct and an unstable sort of them
    yields exactly the stable order (numpy's SIMD sort of 64-bit integers: 2.7x a stable
    argsort on 2M scores). The mapping reproduces the comparison the stable sort makes:
    `-0.0` and `+0.0` get one key (they compare equal), every NaN gets the key of one
    positive quiet NaN, above +inf (numpy sorts NaN last, in index order), and the `+ 0.0`
    that merges the zeros also treats a subnormal as the sort's comparison does under the
    thread's DAZ setting. Anything that is not float32, or longer than 2**32, takes the
    argsort itself.
    """
    s = np.asarray(scores)
    n = s.shape[0]
    if s.dtype != np.float32 or s.ndim != 1 or n >= 2 ** 32:
        return np.argsort(-s, kind="stable")
    v = np.negative(s)
    v += np.float32(0.0)
    nan = np.isnan(v)
    if nan.any():
        v[nan] = np.float32(np.nan)
    b = v.view(np.uint32)
    sign = np.uint32(0x80000000)
    key = np.where((b & sign) != 0, ~b, b | sign).astype(np.uint64)
    key <<= np.uint64(32)
    key |= np.arange(n, dtype=np.uint64)
    key.sort()
    key &= np.uint64(0xFFFFFFFF)
    return key.astype(np.int64)


def select_positives(scores, is_target, thr, max_frac=0.5):
    """`(tda_q(scores, is_target) <= thr) & is_target`, from a top window when it is certified.

    Returns the boolean mask, or None when the window cannot be certified (the caller then
    runs the full `tda_q`). The accepted set of `tda_q` is the prefix of the stable
    descending order up to the LAST position whose FDR `(cd+1)/max(ct,1)` is at or below
    `thr` (q is a reverse running minimum), so only a prefix long enough to contain that
    position needs sorting:
      - the window is every row scoring at least the `need`-th best decoy score, with
        `need = floor(thr * T) + 1` for T targets. All ties at the cut are included, so
        the window is exactly a prefix of the full stable order, sorted here in that order;
      - certificate: past the window every position has at least the window's cd decoys
        and at most T targets, so its FDR is at least `(cd_W + 1) / T`. When that exceeds
        `thr` (checked in the same float64 arithmetic; division is monotone), no later
        position can be accepted and the window's own last accepted position is the
        global one.
    The counts, FDR values and comparison are those of `tda_q`, so the mask is identical.
    NaN scores, too few decoys, a failed certificate or a window above `max_frac` of the
    rows (where the full sort costs little more) return None.
    """
    s = np.asarray(scores)
    tgt = np.asarray(is_target, dtype=bool)
    n = s.shape[0]
    n_t = int(np.count_nonzero(tgt))
    n_d = n - n_t
    if n == 0 or n_t == 0:
        return None
    need = int(thr * n_t) + 1
    if need > n_d or np.isnan(s).any():
        return None
    dec = s[~tgt]
    cut = np.partition(dec, n_d - need)[n_d - need]
    del dec
    win = np.flatnonzero(s >= cut)
    if win.shape[0] > max_frac * n:
        return None
    order = win[desc_order(s[win])]
    t = tgt[order]
    ct = np.cumsum(t, dtype=np.int64)
    cd = np.cumsum(~t, dtype=np.int64)
    if not ((float(cd[-1]) + 1.0) / float(max(n_t, 1)) > thr):
        return None
    ok = (cd + 1) / np.maximum(ct, 1) <= thr
    pos = np.zeros(n, dtype=bool)
    if ok.any():
        last = int(np.flatnonzero(ok)[-1])
        head = order[:last + 1]
        pos[head[t[:last + 1]]] = True
    return pos


def n_targets_at_windowed(scores, is_target, fdr):
    """`n_targets_at`, from `select_positives` when its window is certified."""
    pos = select_positives(scores, np.asarray(is_target) == 1, fdr)
    if pos is None:
        return n_targets_at(scores, is_target, fdr)
    return int(np.count_nonzero(pos))


def _count_at_fdr_sorted(t_sorted, fdr):
    """Targets accepted at `fdr` given the target mask in DESCENDING score order.

    Exactly `n_targets_at`'s result: cumulative target/decoy counts, FDR = (cd+1)/max(ct,1),
    and because the q-value is a reverse running minimum the accepted set is the prefix up to
    the LAST index whose FDR is at or below the threshold. Returns (count, last_index) with
    last_index = -1 when nothing is accepted.
    """
    ct = np.cumsum(t_sorted, dtype=np.int64)
    cd = np.cumsum(~t_sorted, dtype=np.int64)
    ok = (cd + 1) <= fdr * np.maximum(ct, 1)
    if not ok.any():
        return 0, -1
    last = int(np.flatnonzero(ok)[-1])
    return int(ct[last]), last


def n_targets_at_col(col, tgt, fdr, topk=0):
    """`n_targets_at(col, tgt, fdr)` for one column, without sorting all of it.

    With topk > 0, only the top-k scores are sorted (argpartition is O(n)), falling back to
    the full sort when the FDR boundary is not resolved inside the window.

    NOT exact. `argpartition` does not reproduce `kind="stable"` ordering among equal scores
    at the window edge, so cumulative counts can shift there. Measured on 300k rows x 387
    features: 30 of 774 (column, sign) counts differed, all by 1-2, and the selected feature
    was unchanged. 7.2x faster (17.0s -> 2.3s). Off by default for that reason - it only
    chooses the INITIALISATION feature, which the model then retrains away from, so the
    deviation is defensible, but it should be measured end to end before being turned on.
    """
    n = col.shape[0]
    if topk and topk < n:
        idx = np.argpartition(-col, topk - 1)[:topk]
        idx = idx[np.argsort(-col[idx], kind="stable")]
        cnt, last = _count_at_fdr_sorted(tgt[idx], fdr)
        if last < topk - 1:            # boundary resolved inside the window
            return cnt
    order = np.argsort(-col, kind="stable")
    return _count_at_fdr_sorted(tgt[order], fdr)[0]


# Rows of init sample per scan thread before another thread is worth starting: below this
# a column's sort is a few milliseconds and the pool's start-up and GIL hand-offs dominate.
_SCAN_ROWS_PER_THREAD = 20000


def scan_workers(requested, rows, cols):
    """Threads for the init feature scan, capped by the sample size and the column count."""
    return max(1, min(int(requested), int(cols), rows // _SCAN_ROWS_PER_THREAD))


def n_targets_at_many(X, is_target, fdr, topk=0, workers=1, initializer=None):
    """Best (column, sign) by targets accepted at `fdr`, over every column of X.

    Ties resolve toward the lowest column index and sign +1, matching the original nested
    loop, which scanned j ascending with sign +1 before -1 and used a strict `>`.

    With `workers > 1` the columns are counted on a thread pool (numpy's argsort, cumsum and
    fancy indexing release the GIL), one task per column evaluating both signs from the same
    column read. Every count is computed exactly as in the serial loop, and the winner is
    then reduced over (j, sign) in the serial loop's order with the same strict `>`, so the
    result does not depend on the thread count or on task completion order. `initializer`
    runs once per pool thread; the worker passes its flush-to-zero setting there, because a
    new thread does not inherit the main thread's floating-point control state on Windows.
    """
    tgt = np.asarray(is_target).astype(bool)

    def both_signs(j):
        col = np.ascontiguousarray(X[:, j])
        return (n_targets_at_col(col, tgt, fdr, topk=topk),
                n_targets_at_col(-col, tgt, fdr, topk=topk))

    ncol = X.shape[1]
    if workers > 1 and ncol > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(int(workers), ncol),
                                initializer=initializer) as ex:
            counts = list(ex.map(both_signs, range(ncol)))
    else:
        counts = [both_signs(j) for j in range(ncol)]
    best_j, best_sign, best_n = 0, 1, -1
    for j, pair in enumerate(counts):
        for sign, c in zip((1, -1), pair):
            if c > best_n:
                best_n, best_j, best_sign = c, j, sign
    return best_j, best_sign, best_n



# How much larger than free physical memory a feature matrix may be before the
# disk-backed memmap is used at all.
#
# Deliberately greater than 1: the memmap is a LAST RESORT, not a safety margin. Its
# indexed minibatch reads measured about 9x slower than holding the matrix in RAM -- 166
# minutes against roughly 20 on a 4.52 GiB matrix -- whereas a matrix that merely
# overflows physical memory is paged by the operating system, which for the largely
# sequential access this training does is far cheaper than the explicit memmap path.
# Above this ratio the paging itself would thrash and the memmap wins again.
#
# The consequence is deliberate and worth stating: between 1x and 2x free memory the
# rescore relies on the page file. A machine with no page file, or a small fixed one,
# will hit the allocator instead -- set MUMDIA_NN_STREAM_GB explicitly there.
_FREE_MULTIPLIER = 2.0


def available_ram_bytes():
    """Physical memory currently available, or None when it cannot be determined.

    No new dependency: `psutil` is not in the rescore environment and adding it to reach
    one number is not worth the resolver risk. Windows goes through
    `GlobalMemoryStatusEx`, Linux reads `MemAvailable` (which accounts for reclaimable
    cache, unlike MemFree), and anything else returns None so the caller keeps the fixed
    default rather than guessing.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = _MemStatus()
            m.dwLength = ctypes.sizeof(_MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            return int(m.ullAvailPhys)
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 - any failure means "cannot tell", never fatal
        return None
    return None


def auto_stream_threshold_gb(default_gb):
    """How large a feature matrix may be before the disk-backed backend is used.

    The fixed 4 GB this replaces cost 166 minutes on the machine that prompted the
    change: 96 GiB total, 42 free, and a 4.52 GiB matrix that crossed the threshold by
    13% and took the memmap path instead of the roughly 20 minutes it needed in RAM.

    The memmap is treated as a last resort rather than a safety margin, so the threshold
    is `_FREE_MULTIPLIER` times free memory -- above it, not under it. Between 1x and 2x
    the operating system pages, which for this access pattern is much cheaper than the
    memmap; beyond that the paging thrashes and the memmap is the better of two bad
    options. Never returns less than `default_gb`, so a machine too small to benefit
    keeps exactly today's behaviour.

    Returns `(threshold_gb, why)` so the caller can say where the number came from.
    """
    free = available_ram_bytes()
    if not free:
        return default_gb, "default (available memory could not be determined)"
    free_gb = free / 1024 ** 3
    derived = free_gb * _FREE_MULTIPLIER
    if derived <= default_gb:
        return default_gb, f"default ({free_gb:.1f} GiB free is not enough to raise it)"
    return derived, f"{_FREE_MULTIPLIER:g}x the {free_gb:.1f} GiB free"

# Whether this process flushes subnormals to zero; `_fp_thread_init` copies it to a thread.
_FLUSH = {"on": False}


def _fp_thread_init():
    """Give a Python pool thread the main thread's flush-to-zero state.

    FTZ/DAZ live in each thread's MXCSR, and a thread started on Windows begins with the
    default state rather than inheriting its creator's. Numpy work on a pool thread must see
    the same state as the serial code it replaces, or a subnormal operand could round or
    compare differently there.
    """
    if _FLUSH["on"]:
        import torch

        torch.set_flush_denormal(True)


class _TrainConfig:
    """The training hyperparameters, as plain attributes (picklable for a child process)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _keyed_rng(key):
    """A numpy Generator seeded from a tuple of non-negative integers."""
    return np.random.default_rng([int(k) % (2 ** 63) for k in key])


def _build_trainer(torch, cfg, X, stream, y, fold, feat_cols, keyed_shuffle=False):
    """The per-fold trainer over the standardised matrix `X`; returns `run_fold(seed, f)`.

    Built by the serial path in `main` and by each `MUMDIA_NN_PARALLEL` child process from
    the same `cfg`, so both run the same code. `X` is the in-memory matrix or a memmap;
    `stream` selects the accessors the streaming backend uses.
    """
    nn = torch.nn
    TRAIN_SUB = cfg.TRAIN_SUB
    WARM = cfg.WARM
    WARM_EPOCHS = cfg.WARM_EPOCHS
    NEG_RATIO = cfg.NEG_RATIO
    NEG_SELECT = cfg.NEG_SELECT
    MARGIN_FRAC = cfg.MARGIN_FRAC
    ITERS = cfg.ITERS
    EPOCHS = cfg.EPOCHS
    HIDDEN = cfg.HIDDEN
    DROPOUT = cfg.DROPOUT
    LR = cfg.LR
    WD = cfg.WD
    BATCH = cfg.BATCH
    TRAIN_FDR = cfg.TRAIN_FDR
    INIT_FDR_MAX = cfg.INIT_FDR_MAX
    EARLY_STOP = cfg.EARLY_STOP
    EARLY_STOP_TOL = cfg.EARLY_STOP_TOL
    PREGATHER_GB = cfg.PREGATHER_GB
    CLAMP_TINY = cfg.CLAMP_TINY
    FINAL_POOL_SCORE = cfg.FINAL_POOL_SCORE
    SELECT = cfg.SELECT
    GATHER = cfg.GATHER
    DEVICE = cfg.DEVICE
    DEBUG_DENORMALS = cfg.DEBUG_DENORMALS
    SCAN_THREADS = cfg.SCAN_THREADS
    init_sample_limit = cfg.INIT_SAMPLE
    KEYED_SHUFFLE = bool(keyed_shuffle)
    nf = len(feat_cols)
    if stream:
        get = lambda idx: np.ascontiguousarray(X[idx])
        get_col = lambda idx, j: np.asarray(X[idx, j])
    else:
        get = lambda idx: X[idx]
        get_col = lambda idx, j: np.asarray(X[idx, j])
    # A torch view of the in-memory matrix (shared memory, no copy) for `score_idx`. A
    # read-only memmap (a parallel child's) makes torch warn that writes would be
    # undefined; nothing here writes to it.
    if not stream and GATHER == "torch":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X_t = torch.from_numpy(X)
    else:
        X_t = None
    _score_buf = [None]

    class MLP(nn.Module):
        def __init__(self, d_in, hidden, p):
            super().__init__()
            layers, d = [], d_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
                d = h
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_model(train_idx, pos_weight, seed, warm=None, epochs=None, shuffle_key=None):
        """Minibatch train the MLP on `train_idx`.

        The training row set is FIXED for all EPOCHS, so its features are gathered ONCE
        into a contiguous tensor and each minibatch is then a cheap index into that
        tensor. The previous code fancy-indexed the full feature matrix
        (`Xs[idx]` / `mm[idx]`) once per minibatch, which measured ~25% of total runtime
        at production scale. Falls back to the per-batch path when the gathered block
        would exceed MUMDIA_NN_PREGATHER_GB, so the streaming backend keeps its low-RAM
        guarantee on very large pools.
        """
        torch.manual_seed(seed)
        if warm is None:
            m = MLP(nf, HIDDEN, DROPOUT).to(DEVICE)
            opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        else:
            # Warm start: carry both the weights AND the Adam moments forward. Keeping the
            # optimiser matters - a fresh Adam would re-enter its bias-correction warmup
            # every iteration and undo much of the benefit.
            m, opt = warm
        n_ep = EPOCHS if epochs is None else max(1, int(epochs))
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
        idx = np.asarray(train_idx)
        ntr = len(idx)
        pregather = (ntr * nf * 4) <= PREGATHER_GB * 1024 ** 3
        if pregather:
            Xt = torch.from_numpy(np.ascontiguousarray(get(idx))).to(DEVICE)
            yt = torch.from_numpy(np.ascontiguousarray(y[idx])).to(DEVICE)
        for ep_i in range(n_ep):
            m.train()
            # numpy RNG drives the shuffle in both paths, so the training trajectory
            # stays tied to the existing np.random.seed(seed) stream. Under
            # MUMDIA_NN_PARALLEL the stream is keyed per (seed, fold, iteration, epoch)
            # instead, so a fold's trajectory does not depend on the folds before it.
            if shuffle_key is None:
                order = np.random.permutation(ntr)
            else:
                order = _keyed_rng(shuffle_key + (ep_i,)).permutation(ntr)
            # A trailing minibatch of exactly one row makes BatchNorm1d raise "Expected more
            # than 1 value per channel when training" (it cannot compute a batch variance
            # from one sample). ntr changes with the selected positive set every iteration, so
            # this is otherwise a lurking crash that could land in the final experiment-wide
            # rescore, after every run's compute has been spent. Drop that single row for this
            # epoch; the permutation is reshuffled next epoch, so no row is systematically
            # excluded from training.
            n_use = ntr - 1 if (ntr % BATCH) == 1 and ntr > BATCH else ntr
            if pregather:
                perm_t = torch.from_numpy(order).to(DEVICE)
                for i in range(0, n_use, BATCH):
                    b = perm_t[i:i + BATCH]
                    opt.zero_grad()
                    lossf(m(Xt[b]), yt[b]).backward()
                    opt.step()
            else:
                perm = idx[order]
                for i in range(0, n_use, BATCH):
                    b = perm[i:i + BATCH]
                    Xb = torch.from_numpy(get(b)).to(DEVICE)
                    yb = torch.from_numpy(y[b]).to(DEVICE)
                    opt.zero_grad()
                    lossf(m(Xb), yb).backward()
                    opt.step()
            _clamp_tiny(m, CLAMP_TINY, opt)
        return m, opt

    @torch.no_grad()
    def score_idx(m, idx):
        m.eval()
        idx = np.asarray(idx)
        out = np.empty(len(idx), np.float32)
        step = BATCH * 4
        if X_t is None:
            for i in range(0, len(idx), step):
                b = idx[i:i + step]
                out[i:i + len(b)] = m(torch.from_numpy(get(b)).to(DEVICE)).cpu().numpy()
            return out
        # In-memory backend: gather each scoring batch with `torch.index_select` into one
        # buffer that lives for the whole run. `Xs[b]` was a single-threaded numpy fancy
        # index plus a fresh 25 MB allocation per batch (16,384 x 387 float32); the torch
        # gather runs on the intra-op threads and writes the same values into the same
        # shape. The streaming backend keeps the numpy path above, which reads the memmap.
        idx_t = torch.from_numpy(np.ascontiguousarray(idx, dtype=np.int64))
        buf = _score_buf[0]
        if buf is None:
            buf = _score_buf[0] = torch.empty((step, nf), dtype=torch.float32)
        for i in range(0, len(idx), step):
            k = min(step, len(idx) - i)
            xb = buf[:k]
            torch.index_select(X_t, 0, idx_t[i:i + k], out=xb)
            out[i:i + k] = m(xb.to(DEVICE)).cpu().numpy()
        return out

    def run_fold(seed, f):
        """Train fold `f` under seed `seed`; return (held-out rows, their scores).

        The fold body of the original per-seed loop, unchanged. Every fold's work is
        self-contained except the epoch shuffle, which draws from the global numpy stream
        (seeded once per seed by the caller) unless `KEYED_SHUFFLE` keys it per
        (seed, fold, iteration, epoch).
        """
        tr_idx = np.where(fold != f)[0]
        te_idx = np.where(fold == f)[0]
        ytr = y[tr_idx]
        tgt_tr = ytr == 1
        if len(tr_idx) == 0 or len(te_idx) == 0:
            raise RuntimeError(
                f"fold {f} is empty in training or holdout; reduce MUMDIA_NN_FOLDS"
            )
        if not (np.any(ytr == 1) and np.any(ytr == 0)):
            raise RuntimeError(
                f"fold {f} training rows do not contain both targets and decoys"
            )
        # Select the initial feature and sign using this fold's training rows
        # only. The old global selection inspected held-out labels and made
        # the nominal OOF scores optimistic. For very large folds, sample
        # evenly across the deterministic training order rather than taking
        # only the file head.
        _t = time.time()
        sample_n = min(len(tr_idx), init_sample_limit)
        while True:
            if sample_n >= len(tr_idx):
                sample_n, init_idx = len(tr_idx), tr_idx
            else:
                positions = np.linspace(0, len(tr_idx) - 1, sample_n, dtype=np.int64)
                init_idx = tr_idx[positions]
            Xsamp, ysamp = get(init_idx), y[init_idx]
            # One column at a time, both signs from the SAME column read, vectorised
            # over feature blocks (see n_targets_at_many): same counts and tie-breaking
            # as the per-feature scan, ~387x fewer Python-level argsort calls.
            best_j, best_sign, best_n = n_targets_at_many(
                Xsamp, ysamp, TRAIN_FDR, topk=env_i("MUMDIA_NN_INIT_TOPK", 0),
                workers=scan_workers(SCAN_THREADS, len(init_idx), Xsamp.shape[1]),
                initializer=_fp_thread_init,
            )
            if best_n > 0 or sample_n >= len(tr_idx):
                break
            # Nothing passes on this sample. That is a property of the sample size
            # relative to the pool's true fraction, not of the features: a pool that
            # is overwhelmingly false (35M candidates screened, 8M PSMs accepted, a
            # few thousand true) puts too few true rows into a fixed 300k sample for
            # any column to accumulate 100 targets before its first decoy. Rescan on
            # 4x the rows rather than rank the fold by an arbitrary column.
            next_n = min(len(tr_idx), sample_n * 4)
            print(f"  seed {seed} fold {f}: no feature reaches {TRAIN_FDR:.0%} on a "
                  f"{sample_n}-row init sample; rescanning on {next_n} rows", flush=True)
            del Xsamp, ysamp
            sample_n = next_n
        score_tr = (best_sign * get_col(tr_idx, best_j)).astype(np.float32)
        _t = _tick("2_init_feature_scan", _t)
        print(f"  seed {seed} fold {f}: init={feat_cols[best_j]} "
              f"sign{best_sign:+d} ({best_n}@{TRAIN_FDR:.0%} "
              f"on {sample_n} training rows)", flush=True)
        model = None
        optim = None
        prev_pos = None
        used_iters = 0
        score_tr_current = True
        for it in range(ITERS):
            _tsel = time.time()
            pos = (select_positives(score_tr, tgt_tr, TRAIN_FDR)
                   if SELECT == "window" else None)
            if pos is not None and pos.any():
                SELECT_COUNTS["window"] = SELECT_COUNTS.get("window", 0) + 1
            else:
                # The full sort: when the window is not certified, or selects nothing
                # (the bootstrap ladder below needs every q-value).
                SELECT_COUNTS["full"] = SELECT_COUNTS.get("full", 0) + 1
                q = tda_q(score_tr, ytr)
                pos = (q <= TRAIN_FDR) & (ytr == 1)
            if model is None and not np.any(pos) and INIT_FDR_MAX > 0:
                # Bootstrap only. The init feature ranks the whole fold here, and on a
                # pool that is overwhelmingly false no single column may reach the
                # training FDR although the model trained on a looser first selection
                # will. Loosen this one selection in steps up to INIT_FDR_MAX; the next
                # iteration re-selects at TRAIN_FDR on the model's scores as always.
                for fdr in (0.02, 0.05, 0.1):
                    if fdr > INIT_FDR_MAX + 1e-12:
                        break
                    pos = (q <= fdr) & (ytr == 1)
                    if np.any(pos):
                        print(f"  seed {seed} fold {f}: init feature has no target at "
                              f"{TRAIN_FDR:.0%}; bootstrap positives selected at {fdr:.0%} "
                              f"({int(pos.sum())} rows), later iterations use {TRAIN_FDR:.0%}",
                              flush=True)
                        break
            neg = ytr == 0
            if not np.any(pos):
                raise RuntimeError(
                    f"fold {f} selected no positive targets at training FDR "
                    f"{TRAIN_FDR}; use a larger PSM pool or review the feature contract"
                )
            # Convergence on the selected positive set (Percolator's criterion). Exact
            # equality is too strict to ever trigger in practice: dropout plus the
            # retrained-from-scratch model perturbs scores enough that a handful of
            # borderline PSMs flip every iteration forever (measured: it never fired
            # on a 40k pool over 10 iterations). So stop when the CHURN - the symmetric
            # difference as a fraction of the selected set - falls below a tolerance,
            # i.e. the training set has stabilised to within noise.
            # MUMDIA_NN_EARLY_STOP_TOL=0 restores exact-equality; EARLY_STOP=0 disables.
            if model is not None and prev_pos is not None:
                churn = int(np.count_nonzero(pos != prev_pos))
                frac = churn / max(1, int(pos.sum()))
                print(f"  seed {seed} fold {f}: iter {used_iters} positive-set churn "
                      f"{churn} ({frac:.3%} of {int(pos.sum())})", flush=True)
                if EARLY_STOP and frac <= EARLY_STOP_TOL:
                    print(f"  seed {seed} fold {f}: converged after {used_iters} "
                          f"iteration(s) (churn {frac:.3%} <= tol "
                          f"{EARLY_STOP_TOL:.3%}); skipping "
                          f"{ITERS - used_iters} remaining", flush=True)
                    _detail("selection", time.time() - _tsel)
                    break
            prev_pos = pos
            sel = tr_idx[pos | neg]
            sel_pos, sel_neg = int(pos.sum()), int(neg.sum())
            pos_i = tr_idx[pos]
            neg_i = tr_idx[neg]
            if NEG_RATIO > 0 and len(neg_i) > NEG_RATIO * len(pos_i):
                # Cap negatives at NEG_RATIO x the positives selected THIS iteration.
                # Training on every decoy in the fold is ~15-19:1 in practice, so most
                # gradient steps are spent on negatives. `pos_weight` below is recomputed
                # from the capped set, so the loss stays balanced for what is actually
                # trained on.
                #
                # This does NOT touch the FDR: decoys are thinned for TRAINING only, while
                # scoring, target/decoy competition and q-values still use the full pool.
                # It can move the learned boundary, hence a knob rather than a default.
                #
                # NEG_SELECT decides WHICH decoys survive. `random` keeps the shape of the
                # decoy distribution. `margin` keeps the highest-scoring ones, i.e. the
                # only part of that distribution still competing with accepted targets,
                # at the cost of never showing the model the easy bulk. `hybrid` splits
                # the budget between the two.
                keep_n = max(1, int(round(NEG_RATIO * len(pos_i))))
                rs_n = np.random.RandomState(
                    (int(seed) * 7919 + int(f) * 104729 + used_iters * 31) % (2 ** 32)
                )
                if NEG_SELECT == "random":
                    neg_i = rs_n.choice(neg_i, size=min(keep_n, len(neg_i)), replace=False)
                else:
                    s_neg = score_tr[neg]
                    order = (desc_order(s_neg) if SELECT == "window"
                             else np.argsort(-s_neg, kind="stable"))
                    if NEG_SELECT == "margin":
                        take = order[:keep_n]
                    else:
                        k_hard = max(1, int(round(MARGIN_FRAC * keep_n)))
                        hard, rest = order[:k_hard], order[k_hard:]
                        k_rand = min(max(0, keep_n - k_hard), len(rest))
                        rand = (
                            rs_n.choice(rest, size=k_rand, replace=False)
                            if k_rand
                            else np.empty(0, np.int64)
                        )
                        take = np.concatenate([hard, rand]).astype(np.int64)
                    neg_i = neg_i[np.sort(take)]
                sel = np.sort(np.concatenate([pos_i, neg_i]))
                sel_pos, sel_neg = len(pos_i), len(neg_i)
                print(
                    "  seed %s fold %s: negative cap %.2fx (%s) -> %d neg for %d pos"
                    % (seed, f, NEG_RATIO, NEG_SELECT, sel_neg, sel_pos),
                    flush=True,
                )
            if TRAIN_SUB > 0:
                # Stratified subsample of the training rows for THIS iteration.
                # Positives and negatives are thinned by the same factor, so the class
                # balance -- and therefore pos_weight below -- is unchanged and only the
                # number of gradient steps falls. Seeded per (seed, fold, iteration) so a
                # rerun reproduces. Positive SELECTION still runs over the FULL fold, so
                # this trades gradient steps for wall time without narrowing what can be
                # discovered.
                # thin whatever survived the negative cap above
                frac = (
                    TRAIN_SUB
                    if TRAIN_SUB <= 1.0
                    else min(1.0, TRAIN_SUB / max(1, len(sel)))
                )
                rs = np.random.RandomState(
                    (int(seed) * 1000003 + int(f) * 1009 + used_iters) % (2 ** 32)
                )
                kp = max(1, int(round(len(pos_i) * frac)))
                kn = max(1, int(round(len(neg_i) * frac)))
                pos_i = rs.choice(pos_i, size=min(kp, len(pos_i)), replace=False)
                neg_i = rs.choice(neg_i, size=min(kn, len(neg_i)), replace=False)
                sel = np.sort(np.concatenate([pos_i, neg_i]))
                sel_pos, sel_neg = len(pos_i), len(neg_i)
                print(
                    "  seed %s fold %s: train subsample frac=%.4f -> %d rows "
                    "(%d pos / %d neg)"
                    % (seed, f, frac, len(sel), sel_pos, sel_neg),
                    flush=True,
                )
            pw = float(sel_neg) / max(1.0, float(sel_pos))
            _detail("selection", time.time() - _tsel)
            _t = time.time()
            # Warm start reuses the previous iteration's weights and Adam state, so a
            # later iteration only adapts to the changed positive set. WARM_EPOCHS (when
            # set) applies from the SECOND iteration on: the first still needs a full
            # run to get off random initialisation.
            warm_in = (model, optim) if (WARM and model is not None) else None
            ep = None
            if WARM and model is not None and WARM_EPOCHS > 0:
                ep = WARM_EPOCHS
            model, optim = train_model(
                sel, pw, seed, warm=warm_in, epochs=ep,
                shuffle_key=(seed, f, used_iters) if KEYED_SHUFFLE else None,
            )
            _t = _tick("3_train", _t)
            if DEBUG_DENORMALS:
                model.eval()
                _xb = torch.from_numpy(get(tr_idx[:BATCH * 4])).to(DEVICE)
                print(
                    "  seed %d fold %d: subnormals params=%d buffers=%d activations=%d "
                    "adam_moments=%d" % ((seed, f) + _denormal_census(model, _xb, optim)),
                    flush=True,
                )
            if it == ITERS - 1 and not FINAL_POOL_SCORE:
                # The last round's pool scores would feed nothing but the log line
                # below: there is no next selection to make from them. Skipping the
                # pass leaves every score untouched (it neither trains nor draws from an
                # RNG), and saves one full training-pool forward pass per fold.
                score_tr_current = False
            else:
                score_tr = score_idx(model, tr_idx)
                _t = _tick("4_score_pool_per_iter", _t)
            used_iters += 1
        _t = time.time()
        te_scores = score_idx(model, te_idx)
        _t = _tick("5_score_holdout", _t)
        if score_tr_current:
            print(f"  seed {seed} fold {f}: train targets@{TRAIN_FDR:.0%} = "
                  f"{n_targets_at(score_tr, ytr, TRAIN_FDR)}", flush=True)
        else:
            print(f"  seed {seed} fold {f}: held-out targets@{TRAIN_FDR:.0%} = "
                  f"{n_targets_at_windowed(te_scores, y[te_idx], TRAIN_FDR)}",
                  flush=True)
        return te_idx, te_scores
    return run_fold


# Files this run created and removes on exit: the memmap and the parallel side arrays.
_LEFTOVERS = []


def _remove_leftovers():
    """Delete `_LEFTOVERS` that are no longer mapped; keep the rest for a later retry."""
    import gc

    gc.collect()
    for path in list(_LEFTOVERS):
        try:
            if os.path.exists(path):
                os.remove(path)
            _LEFTOVERS.remove(path)
        except OSError:
            pass


# Per-task phases and sub-timers reported by MUMDIA_NN_PARALLEL children, summed. They are
# process seconds that overlap in wall time, so they are printed apart from PHASE.
CHILD_PHASE = {}
CHILD_DETAIL = {}

# State of a MUMDIA_NN_PARALLEL child process, set once by `_parallel_child_init`.
_CHILD = {}


def _exit_with_parent(parent):
    """Stop this child when the parent worker dies, instead of training for nobody.

    The engine kills the worker it spawned on error or interrupt (`ChildGuard`), but not the
    worker's own children; without this they would finish their current fold first.
    """
    parent.join()
    os._exit(3)


def _parallel_child_init(spec):
    """Initialise one fold-training child: threads, flush-to-zero, the shared matrix."""
    import multiprocessing

    import torch

    if spec["threads"] > 0:
        torch.set_num_threads(int(spec["threads"]))
    _FLUSH["on"] = bool(spec["flush"])
    if spec["flush"]:
        torch.set_flush_denormal(True)
    # Read-only and shared: every child maps the same file, so the page cache holds one copy
    # of the matrix however many children read it.
    X = np.memmap(spec["mm_path"], dtype=np.float32, mode="r",
                  shape=(int(spec["n"]), int(spec["nf"])))
    y = np.load(spec["y_path"])
    fold = np.load(spec["fold_path"])
    cfg = spec["cfg"]
    # The init scan's thread pool shares this process's budget; its result does not depend
    # on the thread count.
    cfg.SCAN_THREADS = max(1, min(int(cfg.SCAN_THREADS), int(spec["threads"])))
    _CHILD["run_fold"] = _build_trainer(torch, cfg, X, spec["stream"], y, fold,
                                        spec["feat_cols"], keyed_shuffle=True)
    parent = multiprocessing.parent_process()
    if parent is not None:
        threading.Thread(target=_exit_with_parent, args=(parent,), daemon=True).start()


def _parallel_child_task(seed, f):
    """Train one (seed, fold) in a child; return its held-out scores, timers and log text."""
    import contextlib
    import io

    PHASE.clear()
    DETAIL.clear()
    SELECT_COUNTS.clear()
    log = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(log):
        _te_idx, te_scores = _CHILD["run_fold"](seed, f)
    return (seed, f, te_scores, dict(PHASE), dict(DETAIL), dict(SELECT_COUNTS),
            log.getvalue(), time.time() - t0, os.getpid())


def _run_parallel(spec, seeds, folds, workers, fold):
    """Train every (seed, fold) in `workers` spawned processes; return {(seed, fold): scores}.

    Tasks are independent under the keyed shuffle, so which process runs which task, and in
    what order, cannot change a score. Each child's log is printed in one block when its task
    ends; the results are assembled afterwards in (seed, fold) order.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = [(sd, f) for sd in seeds for f in range(folds)]
    ctx = multiprocessing.get_context("spawn")
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_parallel_child_init, initargs=(spec,))
    results = {}
    futs = []
    try:
        futs = [ex.submit(_parallel_child_task, sd, f) for sd, f in tasks]
        for fu in as_completed(futs):
            sd, f, scores, ph, de, sc, log, wall, pid = fu.result()
            want = int(np.count_nonzero(fold == f))
            if len(scores) != want:
                raise RuntimeError(
                    f"parallel task seed {sd} fold {f} returned {len(scores)} scores for "
                    f"{want} held-out rows")
            results[(sd, f)] = scores
            for k, v in ph.items():
                CHILD_PHASE[k] = CHILD_PHASE.get(k, 0.0) + v
            for k, v in de.items():
                CHILD_DETAIL[k] = CHILD_DETAIL.get(k, 0.0) + v
            for k, v in sc.items():
                SELECT_COUNTS[k] = SELECT_COUNTS.get(k, 0) + v
            sys.stdout.write(log)
            print(f"  seed {sd} fold {f}: trained in child {pid} ({wall:.1f} s)", flush=True)
    except BaseException:
        for fu in futs:
            fu.cancel()
        for proc in list(getattr(ex, "_processes", {}).values()):
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001 - best effort on the way out
                pass
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return results


def _train_in_processes(cfg, X, mm_path, out_path, stream, y, fold, feat_cols, seeds, folds,
                        parallel, side_paths, flush, torch):
    """MUMDIA_NN_PARALLEL: write the per-row arrays next to the memmap and run the tasks."""
    if not isinstance(X, np.memmap) or not mm_path:
        raise RuntimeError("MUMDIA_NN_PARALLEL needs the matrix in a memmap file")
    X.flush()
    n_rows, n_cols = int(X.shape[0]), int(X.shape[1])
    del X
    base = os.path.abspath(out_path)
    y_path, fold_path = base + ".par.y.npy", base + ".par.fold.npy"
    side_paths += [y_path, fold_path]
    np.save(y_path, y)
    np.save(fold_path, fold)
    raw = os.environ.get("MUMDIA_NN_PARALLEL_THREADS", "").strip()
    threads = max(1, int(float(raw))) if raw else max(1, torch.get_num_threads())
    n_tasks = len(seeds) * folds
    workers = max(1, min(int(parallel), n_tasks))
    print(
        "nn_rescore_worker: parallel training: %d process(es) x %d torch thread(s) for "
        "%d task(s) (%d seed(s) x %d fold(s)); the epoch shuffle is keyed per (seed, fold, "
        "iteration, epoch), so the scores differ from the serial default as a seed change "
        "would, and do not depend on the process count" % (workers, threads, n_tasks,
                                                           len(seeds), folds),
        flush=True,
    )
    cpus = os.cpu_count() or 0
    if cpus and workers * threads > cpus:
        print(
            "nn_rescore_worker: %d processes x %d threads oversubscribe the %d visible CPUs; "
            "lower MUMDIA_NN_PARALLEL_THREADS" % (workers, threads, cpus),
            flush=True,
        )
    spec = {
        "cfg": cfg, "mm_path": mm_path, "n": n_rows, "nf": n_cols,
        "stream": bool(stream), "y_path": y_path, "fold_path": fold_path,
        "feat_cols": list(feat_cols), "threads": threads, "flush": bool(flush),
    }
    return _run_parallel(spec, seeds, folds, workers, fold)


def main():
    pin_path, out_path = sys.argv[1], sys.argv[2]

    import torch

    FOLDS = env_i("MUMDIA_NN_FOLDS", 3)
    TRAIN_SUB = env_f("MUMDIA_NN_TRAIN_SUB", 0.0)
    WARM = env_i("MUMDIA_NN_WARM_START", 0) != 0
    WARM_EPOCHS = env_i("MUMDIA_NN_WARM_EPOCHS", 0)
    NEG_RATIO = env_f("MUMDIA_NN_NEG_RATIO", 0.0)
    NEG_SELECT = os.environ.get("MUMDIA_NN_NEG_SELECT", "random").strip().lower()
    if NEG_SELECT not in ("random", "margin", "hybrid"):
        raise ValueError(
            "MUMDIA_NN_NEG_SELECT must be random, margin or hybrid (got %r)" % NEG_SELECT
        )
    MARGIN_FRAC = env_f("MUMDIA_NN_MARGIN_FRAC", 0.5)
    ITERS = env_i("MUMDIA_NN_ITERS", 5)
    EPOCHS = env_i("MUMDIA_NN_EPOCHS", 25)
    HIDDEN = [int(x) for x in os.environ.get("MUMDIA_NN_HIDDEN", "128,64").split(",") if x]
    DROPOUT = env_f("MUMDIA_NN_DROPOUT", 0.3)
    LR = env_f("MUMDIA_NN_LR", 1e-3)
    WD = env_f("MUMDIA_NN_WD", 1e-4)
    BATCH = env_i("MUMDIA_NN_BATCH", 4096)
    TRAIN_FDR = env_f("MUMDIA_NN_TRAIN_FDR", 0.01)
    INIT_FDR_MAX = env_f("MUMDIA_NN_INIT_FDR_MAX", 0.05)
    N_SEEDS = env_i("MUMDIA_NN_SEEDS", 1)
    BASE_SEED = env_i("MUMDIA_NN_SEED", 0)
    CHUNK = env_i("MUMDIA_NN_CHUNK", 250000)
    EARLY_STOP = env_i("MUMDIA_NN_EARLY_STOP", 1) != 0
    EARLY_STOP_TOL = env_f("MUMDIA_NN_EARLY_STOP_TOL", 0.01)
    PREGATHER_GB = env_f("MUMDIA_NN_PREGATHER_GB", 8)
    CLAMP_TINY = env_f("MUMDIA_NN_CLAMP_TINY", 1e-20)
    FINAL_POOL_SCORE = env_i("MUMDIA_NN_FINAL_POOL_SCORE", 0) != 0
    # Opt-in: train the (seed, fold) tasks in this many spawned processes at once, with the
    # epoch shuffle keyed per task. 0 (default) is the serial loop and today's scores.
    PARALLEL = max(0, env_i("MUMDIA_NN_PARALLEL", 0))
    SELECT = os.environ.get("MUMDIA_NN_SELECT", "window").strip().lower()
    if SELECT not in ("window", "full"):
        raise ValueError("MUMDIA_NN_SELECT must be window or full (got %r)" % SELECT)
    GATHER = os.environ.get("MUMDIA_NN_GATHER", "torch").strip().lower()
    if GATHER not in ("torch", "numpy"):
        raise ValueError("MUMDIA_NN_GATHER must be torch or numpy (got %r)" % GATHER)
    # auto (default) uses the GPU when torch can see one; cuda/cpu force it. Forcing is
    # what makes a device-only comparison possible: same environment, same package
    # versions, same data, only the device differs (CUDA_VISIBLE_DEVICES="" does NOT
    # reliably hide the GPU from torch).
    _dev = os.environ.get("MUMDIA_NN_DEVICE", "auto").strip().lower()
    if _dev not in ("auto", "cuda", "cpu"):
        raise ValueError("MUMDIA_NN_DEVICE must be auto, cuda or cpu (got %r)" % _dev)
    if _dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "MUMDIA_NN_DEVICE=cuda but torch reports no CUDA device; this torch build is "
            "%s -- install a CUDA build to use the GPU" % torch.__version__
        )
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
    if FOLDS < 2 or ITERS < 1 or EPOCHS < 1 or N_SEEDS < 1:
        raise ValueError("folds>=2, iterations>=1, epochs>=1, and seeds>=1 are required")

    # Torch CPU threads. Measured: the tiny MLP saturates at ~16 threads and is SLOWER at
    # 32 (oversubscription on small GEMMs), so cap rather than inherit all cores.
    #
    # Precedence: an EXPLICIT MUMDIA_NN_THREADS wins over OMP_NUM_THREADS. It used to be the
    # other way round, which meant a site-wide OMP_NUM_THREADS=1 - a common cluster module
    # default - silently pinned the rescore to one thread with no way to override it from
    # MuMDIA's own knob. OMP_NUM_THREADS is still honoured when MUMDIA_NN_THREADS is unset, so
    # a caller who sets only the generic variable still gets what they asked for.
    if DEVICE == "cpu":
        if "MUMDIA_NN_THREADS" in os.environ:
            want = env_i("MUMDIA_NN_THREADS", 16)
            src = "MUMDIA_NN_THREADS"
        elif "OMP_NUM_THREADS" in os.environ:
            want = env_i("OMP_NUM_THREADS", 16)
            src = "OMP_NUM_THREADS"
        else:
            want, src = 16, "default"
        # The engine's --threads sizes its own rayon pool for the whole machine; for the
        # MLP it is an upper bound, not a target. See `performance_cores`.
        cap, cap_why = torch_thread_cap()
        use = want if (want <= 0 or cap <= 0) else min(want, cap)
        if use > 0:
            torch.set_num_threads(max(1, min(use, os.cpu_count() or use)))
        print(
            "nn_rescore_worker: torch cpu threads=%d (asked %d from %s; cap %d: %s; "
            "%d cores visible)"
            % (torch.get_num_threads(), want, src, cap, cap_why, os.cpu_count() or -1),
            flush=True,
        )
    # Subnormal float32 values are handled by microcode assists on Intel cores, at roughly
    # a hundred times the cost of a normal FMA. Measured on an i9-13900KS: one 16384 x 387
    # Linear takes 4.2 ms with normal inputs, 475 ms with subnormal ones, 150 ms with
    # subnormal weights, and 3.5 ms again with flush-to-zero; an EPYC 9354 pays nothing
    # either way (12.1 against 11.4 ms). The subnormals are the first-layer weights of the
    # constant feature columns (see `constant_columns`), which is why the same six-run pool
    # took 118 minutes on that desktop and 19 on the fleet while single-threaded epoch
    # speed was identical. Dropping those columns (`constant_columns`) removes that source
    # but not the others: dead hidden units and their BatchNorm running variances decay the
    # same way (census on the six-run pool: up to ~6,400 parameters, 11 buffers, ~3,000
    # activations in a round), and without flushing the desktop still took 61.8 min against
    # 11.4 with it. Flushing perturbs the arithmetic below 1.2e-38, and any such perturbation
    # reshuffles a single seed's count by up to ~0.4% -- changing the thread count or the
    # column set does the same (116,405 -> 116,192 at 8 threads, -> 116,025 with the constant
    # columns dropped, -> 115,937 flushed, all seed 0) -- while the mean over seeds is flat
    # (Astral, flushed, three seeds: -0.24 / -0.04 / +0.03%).
    FLUSH_DENORMAL = env_i("MUMDIA_NN_FLUSH_DENORMAL", 1) != 0
    if FLUSH_DENORMAL:
        _ftz = torch.set_flush_denormal(True)
        print(
            "nn_rescore_worker: flush subnormal floats to zero: %s"
            % ("on" if _ftz else "unsupported on this CPU"),
            flush=True,
        )
    DEBUG_DENORMALS = env_i("MUMDIA_NN_DEBUG_DENORMALS", 0) != 0

    _FLUSH["on"] = FLUSH_DENORMAL

    # The parquet in-memory load: threads that fill the matrix (0 = the original serial loop,
    # with no read-ahead and no pre_buffer), a one-deep row-group read-ahead, and pre_buffer.
    # auto is at most 8: each fill thread holds a 32,768-row float64 moment buffer (0.1 GB at
    # 387 features), and the fill is memory-bound well before 8 threads.
    _load_raw = os.environ.get("MUMDIA_NN_LOAD_THREADS", "auto").strip().lower()
    LOAD_THREADS = (
        min(8, torch.get_num_threads())
        if _load_raw in ("", "auto")
        else max(0, int(float(_load_raw)))
    )
    READ_AHEAD = env_i("MUMDIA_NN_READ_AHEAD", 1) != 0
    PRE_BUFFER = env_i("MUMDIA_NN_PRE_BUFFER", 1) != 0
    # Threads for the worker's own numpy thread pools (the init scan here). Defaults to the
    # torch CPU thread count resolved above, which already carries the cap.
    _scan_raw = os.environ.get("MUMDIA_NN_SCAN_THREADS", "auto").strip().lower()
    SCAN_THREADS = (
        min(torch.get_num_threads(), _THREAD_CAP)
        if _scan_raw in ("", "auto")
        else max(1, int(float(_scan_raw)))
    )

    stream_env = os.environ.get("MUMDIA_NN_STREAM", "auto").lower()
    filesize = os.path.getsize(pin_path)
    if pin_path.lower().endswith((".parquet", ".pq")):
        # Compare DECODED bytes, not compressed-on-disk bytes: a column store is several
        # times smaller than the equivalent text, so the raw file size would understate the
        # memory a full read actually needs.
        _md = pq.read_metadata(pin_path)
        # Count the feature columns by name. Subtracting a hardcoded 3 undercounted the
        # non-feature columns, of which NON_FEATURE lists 7, so the estimate that picks the
        # backend was biased upward and could stream a matrix that fits (docs/31 F5).
        _nf_guess = max(1, sum(1 for c in pq.read_schema(pin_path).names if c not in NON_FEATURE))
        filesize = int(_md.num_rows) * _nf_guess * 4
    # An explicit MUMDIA_NN_STREAM_GB wins; otherwise size it from free memory, because
    # a fixed 4 GB is both too small on a workstation and too large on a laptop.
    if os.environ.get("MUMDIA_NN_STREAM_GB"):
        stream_gb, stream_why = env_f("MUMDIA_NN_STREAM_GB", 4), "MUMDIA_NN_STREAM_GB"
    else:
        stream_gb, stream_why = auto_stream_threshold_gb(4)
    stream = stream_env in ("1", "on", "true") or (
        stream_env == "auto" and filesize > stream_gb * 1024 ** 3
    )
    # Say WHY a backend was chosen. The auto-threshold is on PIN *text* size, so crossing
    # ~1M PSMs silently switched to the disk-backed memmap and started requiring a
    # writable path -- an invisible change of behaviour when it went wrong.
    print(
        f"nn_rescore_worker: PIN {filesize / 1024 ** 3:.2f} GB, threshold {stream_gb:.2f} GB "
        f"({stream_why}), MUMDIA_NN_STREAM={stream_env} -> "
        f"backend={'stream(memmap)' if stream else 'in-memory'}"
        f", format={'parquet' if pin_path.lower().endswith(('.parquet', '.pq')) else 'tsv'}",
        flush=True,
    )

    if stream and stream_env == "auto":
        print(
            f"nn_rescore_worker: the {filesize / 1024 ** 3:.2f} GB feature matrix exceeds the "
            f"{stream_gb:.2f} GB threshold, so scoring runs from a disk-backed memmap. That is "
            "correct but MUCH slower than holding it in RAM. Raise MUMDIA_NN_STREAM_GB if the "
            "machine has the memory, or set rescore.feature_preset = compact to shrink the "
            "matrix.",
            flush=True,
        )

    _t = time.time()
    # Parquet or the legacy tab-separated PIN, decided by extension.
    IS_PQ = pin_path.lower().endswith((".parquet", ".pq"))
    if IS_PQ:
        _sch = pq.read_schema(pin_path)
        header = list(_sch.names)
    else:
        header = pd.read_csv(pin_path, sep=chr(9), nrows=0).columns.tolist()
    feat_cols = [c for c in header if c not in NON_FEATURE]
    # Optional feature subset. Applied HERE, before either backend reads the PIN, so the
    # dropped columns are never parsed, never standardised and never moved -- which is the
    # point, since at scale the cost is data movement rather than arithmetic. Accepts a
    # comma-separated list of names or a path to a file with one name per line. The ranking
    # is deliberately NOT computed here: the caller chooses the subset, so an experiment can
    # sweep any selection criterion without the worker taking a position on importance.
    _all_feats = list(feat_cols)
    want = os.environ.get("MUMDIA_NN_FEATURES", "").strip()
    if want:
        if os.path.exists(want):
            names = [ln.strip() for ln in open(want, encoding="utf-8") if ln.strip()]
        else:
            names = [x.strip() for x in want.split(",") if x.strip()]
        missing = [x for x in names if x not in _all_feats]
        if missing:
            raise ValueError(
                "MUMDIA_NN_FEATURES lists %d column(s) absent from the PIN, first few: %s"
                % (len(missing), missing[:5])
            )
        keepset = set(names)
        # Keep PIN order, so the selection is a projection and not a reordering.
        feat_cols = [c for c in _all_feats if c in keepset]
        if not feat_cols:
            raise ValueError("MUMDIA_NN_FEATURES resolved to an empty feature list")
        print(
            "nn_rescore_worker: feature subset active -- %d of %d features"
            % (len(feat_cols), len(_all_feats)),
            flush=True,
        )
    # Constant columns never train. Standardised to exactly 0 they contribute nothing to any
    # prediction, but their first-layer weights receive only Adam's L2 term and decay below
    # 1.2e-38 within a few thousand steps (measured census: about 10 columns x 128 units on
    # the six-run Astral pool). Every FMA on a subnormal operand then takes a microcode
    # assist on Intel cores, ~100x a normal one, which made a desktop rescore 6x slower than
    # the same pool on an EPYC. Dropping them here removes the source without touching the
    # arithmetic of anything else; the parquet footer identifies them, so nothing is read.
    if IS_PQ and env_i("MUMDIA_NN_DROP_CONSTANT", 1) != 0:
        _const = constant_columns(pin_path, feat_cols)
        if _const and len(_const) < len(feat_cols):
            _drop = set(_const)
            feat_cols = [c for c in feat_cols if c not in _drop]
            print(
                "nn_rescore_worker: dropped %d constant feature column(s): %s%s"
                % (len(_const), ", ".join(_const[:6]), ", ..." if len(_const) > 6 else ""),
                flush=True,
            )
    nf = len(feat_cols)
    if nf == 0:
        raise ValueError("PIN contains no rescoring feature columns")
    # Cross-validation fold assignment.
    #
    # The engine writes a fold key per PIN row (`base_peptide_id`) and names the file in
    # MUMDIA_NN_FOLD_KEYS. Use it when present: a target and its paired decoy share
    # base_peptide_id, so they land in the same fold, which is what `percolator_lite` does
    # and what docs/11 claims this worker does.
    #
    # The fallback hashes the mod-stripped peptidoform, which does NOT pair them: the
    # DECOY_ marker survives `strip_pep`, so `DECOY_PEPTIDE` and `PEPTIDE` hash apart, and
    # for a reverse-decoy library the decoy peptidoform is the reversed sequence and no
    # string derived from it can reach its target at all. Cross-fold memorisation of a
    # peptide then depresses its own pair, which is conservative but costs sensitivity and
    # makes the fold split depend on the decoy recipe.
    fold_keys = None
    _fk_path = os.environ.get("MUMDIA_NN_FOLD_KEYS", "")
    if _fk_path and os.path.exists(_fk_path):
        try:
            fold_keys = pq.read_table(_fk_path).column("fold_key").to_numpy()
        except Exception as exc:  # pragma: no cover - unreadable companion file
            print(
                "nn_rescore_worker: could not read MUMDIA_NN_FOLD_KEYS (%s: %s); "
                "falling back to the peptidoform hash, which does not pair a target "
                "with its decoy" % (_fk_path, exc),
                flush=True,
            )
            fold_keys = None
    if fold_keys is None:
        print(
            "nn_rescore_worker: no fold-key file; folding on the peptidoform hash, which "
            "does not place a target and its paired decoy in the same fold",
            flush=True,
        )

    _folds = lambda peptides, off=0: folds_for(peptides, fold_keys, FOLDS, off)

    mm_path = None
    if not stream:
        # ---- in-memory backend (median/IQR standardisation) ----
        if IS_PQ:
            # ---- Parquet in-memory: one float32 matrix, standardised in place ----
            # Metadata columns first; they are small as long as they stay Arrow. The
            # previous `to_pylist()` round trips made a Python string per row out of SpecId
            # and Peptide (~0.7 GB on a 3M-row pool) that then sat in the allocator's
            # high-water mark for the whole run. Peptide is only needed when there is no
            # fold-key table to fold on.
            _cols = ["SpecId", "Label"] + ([] if fold_keys is not None else ["Peptide"])
            _tb = pq.read_table(pin_path, columns=_cols)
            y = (_tb.column("Label").to_numpy() == 1).astype(np.float32)
            n = len(y)
            cids = _row_ids(_tb.column("SpecId"))
            if fold_keys is not None:
                fold = _folds(range(n))
            else:
                fold = _folds(_tb.column("Peptide").to_pylist())
            del _tb
            if PARALLEL > 0:
                # The children map this file read-only instead of each copying the matrix.
                mm_path = os.path.abspath(out_path + ".feat.mm")
                _LEFTOVERS.append(mm_path)
                Xs = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(n, nf))
            else:
                Xs = np.empty((n, nf), np.float32)
            if LOAD_THREADS == 0:
                s1, s2 = _fill_parquet_matrix_legacy(pin_path, feat_cols, n, CHUNK, Xs)
            else:
                s1, s2 = fill_parquet_matrix(
                    pin_path, feat_cols, n, CHUNK, Xs, threads=LOAD_THREADS,
                    read_ahead=READ_AHEAD, pre_buffer=PRE_BUFFER,
                    initializer=_fp_thread_init,
                )
            # The decoded Arrow buffers are garbage now; hand them back before training
            # starts, or they stay in RSS for the whole run.
            _release_allocator_slack()
            _ts = time.time()
            mean, std = moments_to_mean_std(s1, s2, n)
            standardise_matrix(Xs, mean, std, CHUNK, threads=LOAD_THREADS,
                               initializer=_fp_thread_init)
            _detail("load: standardise", time.time() - _ts)
        else:
            pin = pd.read_csv(pin_path, sep=chr(9))
        if not IS_PQ:
            y = (pin["Label"].to_numpy() == 1).astype(np.float32)
            cids = np.array(
                [int(s.rsplit("_", 1)[-1]) for s in pin["SpecId"].astype(str)], np.int64
            )
            fold = _folds(pin["Peptide"].tolist())
            X = np.nan_to_num(
                pin[feat_cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            del pin
            # Mean/std, the same transform as the parquet and streaming backends.
            #
            # This path used median/IQR while the other two used mean/std, so the same PSM
            # pool produced different scores depending on `rescore.handoff` and on which
            # side of MUMDIA_NN_STREAM_GB the matrix fell: a 3.99 GB PIN was standardised
            # one way and a 4.01 GB PIN the other, with nothing in the log to say which
            # (docs/31 F5). Converging on mean/std leaves the shipped default (parquet)
            # and every published benchmark unchanged, and only moves this legacy path.
            mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = X.std(axis=0, dtype=np.float64).astype(np.float32)
            std[std == 0] = 1.0
            Xs = np.clip((X - mean) / std, -8, 8).astype(np.float32)
            del X
            n = len(y)
            if PARALLEL > 0:
                mm_path = os.path.abspath(out_path + ".feat.mm")
                _LEFTOVERS.append(mm_path)
                _mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=Xs.shape)
                _mm[:] = Xs
                Xs = _mm
                del _mm
    else:
        # ---- streaming memmap backend (mean/std, one text pass) ----
        if IS_PQ:
            n = int(pq.read_metadata(pin_path).num_rows)
        else:
            with open(pin_path, "rb") as fh:
                n = sum(1 for _ in fh) - 1
        # Resolve ABSOLUTELY: this used to be a path relative to the caller's cwd, so a
        # rescore launched from a different directory failed with a bare
        # `OSError: [Errno 22]` naming a path it could not create.
        mm_path = os.path.abspath(out_path + ".feat.mm")
        _LEFTOVERS.append(mm_path)
        mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(n, nf))
        y = np.empty(n, np.float32)
        cids = np.empty(n, np.int64)
        fold = np.empty(n, np.int16)
        s1 = np.zeros(nf, np.float64)
        s2 = np.zeros(nf, np.float64)
        keep = set(["SpecId", "Label", "Peptide"] + feat_cols)
        off = 0
        def _chunks():
            "Yield fixed-size frames from either backing format."
            if IS_PQ:
                pf = pq.ParquetFile(pin_path)
                cols = ["SpecId", "Label", "Peptide"] + feat_cols
                for batch in pf.iter_batches(batch_size=CHUNK, columns=cols):
                    yield batch.to_pandas()
            else:
                for ch in pd.read_csv(pin_path, sep=chr(9),
                                      usecols=lambda c: c in keep, chunksize=CHUNK):
                    yield ch

        for chunk in _chunks():
            k = len(chunk)
            xf = np.nan_to_num(chunk[feat_cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mm[off:off + k] = xf
            _accumulate_moments(xf, s1, s2)
            y[off:off + k] = (chunk["Label"].to_numpy() == 1).astype(np.float32)
            cids[off:off + k] = [int(s.rsplit("_", 1)[-1]) for s in chunk["SpecId"].astype(str)]
            fold[off:off + k] = _folds(chunk["Peptide"].tolist(), off)
            off += k
        mean = (s1 / n).astype(np.float32)
        std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)
        std[std == 0] = 1.0
        # standardise the memmap in place, chunked (binary, sequential, low RAM)
        for i in range(0, n, CHUNK):
            mm[i:i + CHUNK] = np.clip((mm[i:i + CHUNK] - mean) / std, -8, 8)
        mm.flush()
    _t = _tick("1_pin_read_standardise", _t)
    init_sample_limit = env_i("MUMDIA_NN_INIT_SAMPLE", 300000)
    print(f"nn_rescore_worker: device={DEVICE} backend={'stream' if stream else 'in-memory'} "
          f"pool={n} feats={nf}", flush=True)

    cfg = _TrainConfig(
        FOLDS=FOLDS, TRAIN_SUB=TRAIN_SUB, WARM=WARM, WARM_EPOCHS=WARM_EPOCHS,
        NEG_RATIO=NEG_RATIO, NEG_SELECT=NEG_SELECT, MARGIN_FRAC=MARGIN_FRAC, ITERS=ITERS,
        EPOCHS=EPOCHS, HIDDEN=HIDDEN, DROPOUT=DROPOUT, LR=LR, WD=WD, BATCH=BATCH,
        TRAIN_FDR=TRAIN_FDR, INIT_FDR_MAX=INIT_FDR_MAX, EARLY_STOP=EARLY_STOP,
        EARLY_STOP_TOL=EARLY_STOP_TOL, PREGATHER_GB=PREGATHER_GB, CLAMP_TINY=CLAMP_TINY,
        FINAL_POOL_SCORE=FINAL_POOL_SCORE, SELECT=SELECT, GATHER=GATHER, DEVICE=DEVICE,
        DEBUG_DENORMALS=DEBUG_DENORMALS, SCAN_THREADS=SCAN_THREADS,
        INIT_SAMPLE=init_sample_limit,
    )
    X = mm if stream else Xs
    side_paths = _LEFTOVERS
    run_fold = None
    try:
        seeds = list(range(BASE_SEED, BASE_SEED + N_SEEDS))
        if PARALLEL > 0:
            _t = time.time()
            by_task = _train_in_processes(
                cfg, X, mm_path, out_path, stream, y, fold, feat_cols, seeds, FOLDS,
                PARALLEL, side_paths, FLUSH_DENORMAL, torch,
            )
            _t = _tick("6_parallel_folds_wall", _t)
        else:
            run_fold = _build_trainer(torch, cfg, X, stream, y, fold, feat_cols)

        # seed ensemble: average rank-normalised out-of-fold scores across seeds
        acc = np.zeros(n, np.float64)
        for s in seeds:
            oof = np.zeros(n, np.float32)
            if PARALLEL > 0:
                for f in range(FOLDS):
                    oof[np.where(fold == f)[0]] = by_task[(s, f)]
            else:
                np.random.seed(s)
                torch.manual_seed(s)
                for f in range(FOLDS):
                    te_idx, te_scores = run_fold(s, f)
                    oof[te_idx] = te_scores
            acc += pd.Series(oof).rank(method="average").to_numpy() / n
        final = acc / N_SEEDS

        out = pa.table({
            "candidate_id": pa.array(cids.astype(np.uint32), pa.uint32()),
            "score": pa.array(final.astype(np.float64), pa.float64()),
            "q_value": pa.array(np.zeros(n, np.float64), pa.float64()),
        })
        pq.write_table(out, out_path)
    finally:
        # Drop every reference to the memmap (the trainer's torch view included) before
        # removing its file: Windows refuses to delete a file that is still mapped. On an
        # error the traceback still references it; `_entry` retries once that is released.
        mm = Xs = X = run_fold = None
        _remove_leftovers()
    _print_timers()
    print(f"nn_rescore_worker: {n} PSMs rescored (targets+decoys), {N_SEEDS} seed(s), "
          f"OOF at {FOLDS} folds, backend={'stream' if stream else 'in-memory'}", flush=True)


def _entry():
    """Run `main`, then remove this run's files even when it failed.

    The memmap and the parallel side arrays are useless once the worker exits. On an error
    the in-flight traceback's frames still reference the matrix, and Windows refuses to
    delete a mapped file, so the removal is retried here, outside the handler, after the
    traceback has been printed and released. Exit codes and the printed traceback are those
    of an uncaught exception.
    """
    code = None
    try:
        main()
    except SystemExit as exc:
        code = exc.code
    except BaseException:  # noqa: BLE001 - reported below, then the worker exits nonzero
        import traceback

        traceback.print_exc()
        code = 1
    _remove_leftovers()
    if code is not None:
        sys.exit(code)


if __name__ == "__main__":
    _entry()
