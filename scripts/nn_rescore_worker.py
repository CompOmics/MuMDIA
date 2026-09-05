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
      feature matrix is held in RAM (median/IQR standardisation).
    - streaming memmap (large PINs, or MUMDIA_NN_STREAM=1): the PIN is read ONCE in
      chunks into a disk-backed float32 memmap (mean/std standardisation accumulated
      in the same pass); training and scoring then draw MINIBATCHES indexed into the
      memmap, so peak RAM is one batch + per-row metadata, NOT the whole matrix.
      This is what makes combining many runs into one rescoring tractable: the full
      PIN never lives in RAM at once.

Determinism note (plan.md Section 7): NN training is only approximately reproducible.
Set MUMDIA_NN_SEEDS>1 to ensemble seeds and average out-of-fold scores.

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
    MUMDIA_NN_STREAM_GB   = 4        auto-stream when the PIN exceeds this many GB
    MUMDIA_NN_CHUNK       = 250000   PIN rows per read chunk (streaming backend)
    MUMDIA_NN_INIT_SAMPLE = 300000   rows used to pick the init feature (streaming)
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
    MUMDIA_NN_THREADS     = 16       torch CPU threads (0 = leave torch's default)
    MUMDIA_NN_PREGATHER_GB= 8        pre-gather the fold's training rows when they fit in
                                     this many GB (one gather per iteration instead of a
                                     fancy-index copy per minibatch)

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
import time

import numpy as np
import pandas as pd
import pyarrow as pa
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


PHASE = {}


def _tick(name, t0):
    "Accumulate elapsed wall time under `name` and return a fresh timestamp."
    PHASE[name] = PHASE.get(name, 0.0) + (time.time() - t0)
    return time.time()


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


def n_targets_at_many(X, is_target, fdr, topk=0):
    """Best (column, sign) by targets accepted at `fdr`, over every column of X.

    Ties resolve toward the lowest column index and sign +1, matching the original nested
    loop, which scanned j ascending with sign +1 before -1 and used a strict `>`.
    """
    tgt = np.asarray(is_target).astype(bool)
    best_j, best_sign, best_n = 0, 1, -1
    for j in range(X.shape[1]):
        col = np.ascontiguousarray(X[:, j])
        for sign in (1, -1):
            c = n_targets_at_col(col if sign > 0 else -col, tgt, fdr, topk=topk)
            if c > best_n:
                best_n, best_j, best_sign = c, j, sign
    return best_j, best_sign, best_n


def main():
    pin_path, out_path = sys.argv[1], sys.argv[2]

    import torch
    import torch.nn as nn

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
    N_SEEDS = env_i("MUMDIA_NN_SEEDS", 1)
    BASE_SEED = env_i("MUMDIA_NN_SEED", 0)
    CHUNK = env_i("MUMDIA_NN_CHUNK", 250000)
    EARLY_STOP = env_i("MUMDIA_NN_EARLY_STOP", 1) != 0
    EARLY_STOP_TOL = env_f("MUMDIA_NN_EARLY_STOP_TOL", 0.01)
    PREGATHER_GB = env_f("MUMDIA_NN_PREGATHER_GB", 8)
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
        if want > 0:
            torch.set_num_threads(max(1, min(want, os.cpu_count() or want)))
        print(
            "nn_rescore_worker: torch cpu threads=%d (from %s; %d cores visible)"
            % (torch.get_num_threads(), src, os.cpu_count() or -1),
            flush=True,
        )

    stream_env = os.environ.get("MUMDIA_NN_STREAM", "auto").lower()
    filesize = os.path.getsize(pin_path)
    if pin_path.lower().endswith((".parquet", ".pq")):
        # Compare DECODED bytes, not compressed-on-disk bytes: a column store is several
        # times smaller than the equivalent text, so the raw file size would understate the
        # memory a full read actually needs.
        _md = pq.read_metadata(pin_path)
        _nf_guess = max(1, _md.num_columns - 3)      # minus SpecId / Label / Peptide
        filesize = int(_md.num_rows) * _nf_guess * 4
    stream_gb = env_f("MUMDIA_NN_STREAM_GB", 4)
    stream = stream_env in ("1", "on", "true") or (
        stream_env == "auto" and filesize > stream_gb * 1024 ** 3
    )
    # Say WHY a backend was chosen. The auto-threshold is on PIN *text* size, so crossing
    # ~1M PSMs silently switched to the disk-backed memmap and started requiring a
    # writable path -- an invisible change of behaviour when it went wrong.
    print(
        f"nn_rescore_worker: PIN {filesize / 1024 ** 3:.2f} GB, threshold {stream_gb:.2f} GB, "
        f"MUMDIA_NN_STREAM={stream_env} -> backend={'stream(memmap)' if stream else 'in-memory'}"
        f", format={'parquet' if pin_path.lower().endswith(('.parquet', '.pq')) else 'tsv'}",
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
    nf = len(feat_cols)
    if nf == 0:
        raise ValueError("PIN contains no rescoring feature columns")
    fold_of = lambda p: int(hashlib.md5(strip_pep(p).encode()).hexdigest(), 16) % FOLDS

    mm_path = None
    if not stream:
        # ---- in-memory backend (median/IQR standardisation) ----
        if IS_PQ:
            # ---- Parquet in-memory: one float32 matrix, standardised in place ----
            # Metadata columns first; they are small.
            _tb = pq.read_table(pin_path, columns=["SpecId", "Label", "Peptide"])
            y = (_tb.column("Label").to_numpy() == 1).astype(np.float32)
            _spec = _tb.column("SpecId").to_pylist()
            cids = np.array([int(x.rsplit("_", 1)[-1]) for x in _spec], np.int64)
            fold = np.array([fold_of(x) for x in _tb.column("Peptide").to_pylist()], np.int16)
            del _tb, _spec
            n = len(y)
            Xs = np.empty((n, nf), np.float32)
            s1 = np.zeros(nf, np.float64)
            s2 = np.zeros(nf, np.float64)
            _pf = pq.ParquetFile(pin_path)
            off = 0
            for _b in _pf.iter_batches(batch_size=CHUNK, columns=feat_cols):
                k = _b.num_rows
                blk = np.empty((k, nf), np.float32)
                for j in range(nf):
                    blk[:, j] = _b.column(j).to_numpy(zero_copy_only=False)
                np.nan_to_num(blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                Xs[off:off + k] = blk
                s1 += blk.sum(axis=0, dtype=np.float64)
                s2 += (blk.astype(np.float64) ** 2).sum(axis=0)
                off += k
                del blk
            if off != n:
                raise RuntimeError(f"parquet row mismatch: metadata {n}, features {off}")
            mean = (s1 / n).astype(np.float32)
            std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)
            std[std == 0] = 1.0
            for i in range(0, n, CHUNK):
                Xs[i:i + CHUNK] = np.clip((Xs[i:i + CHUNK] - mean) / std, -8, 8)
            get = lambda idx: Xs[idx]
            get_col = lambda idx, j: np.asarray(Xs[idx, j])
        else:
            pin = pd.read_csv(pin_path, sep=chr(9))
        if not IS_PQ:
            y = (pin["Label"].to_numpy() == 1).astype(np.float32)
            cids = np.array(
                [int(s.rsplit("_", 1)[-1]) for s in pin["SpecId"].astype(str)], np.int64
            )
            fold = pin["Peptide"].map(fold_of).to_numpy()
            X = np.nan_to_num(
                pin[feat_cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            del pin
            med = np.median(X, axis=0)
            iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
            iqr[iqr == 0] = 1.0
            Xs = np.clip((X - med) / iqr, -8, 8).astype(np.float32)
            del X
            n = len(y)
            get = lambda idx: Xs[idx]
            get_col = lambda idx, j: np.asarray(Xs[idx, j])
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
            s1 += xf.sum(axis=0, dtype=np.float64)
            s2 += (xf.astype(np.float64) ** 2).sum(axis=0)
            y[off:off + k] = (chunk["Label"].to_numpy() == 1).astype(np.float32)
            cids[off:off + k] = [int(s.rsplit("_", 1)[-1]) for s in chunk["SpecId"].astype(str)]
            fold[off:off + k] = chunk["Peptide"].map(fold_of).to_numpy()
            off += k
        mean = (s1 / n).astype(np.float32)
        std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)).astype(np.float32)
        std[std == 0] = 1.0
        # standardise the memmap in place, chunked (binary, sequential, low RAM)
        for i in range(0, n, CHUNK):
            mm[i:i + CHUNK] = np.clip((mm[i:i + CHUNK] - mean) / std, -8, 8)
        mm.flush()
        get = lambda idx: np.ascontiguousarray(mm[idx])
        get_col = lambda idx, j: np.asarray(mm[idx, j])
    _t = _tick("1_pin_read_standardise", _t)
    init_sample_limit = env_i("MUMDIA_NN_INIT_SAMPLE", 300000)
    print(f"nn_rescore_worker: device={DEVICE} backend={'stream' if stream else 'in-memory'} "
          f"pool={n} feats={nf}", flush=True)

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

    def train_model(train_idx, pos_weight, seed, warm=None, epochs=None):
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
        for _ in range(n_ep):
            m.train()
            # numpy RNG drives the shuffle in both paths, so the training trajectory
            # stays tied to the existing np.random.seed(seed) stream.
            order = np.random.permutation(ntr)
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
        return m, opt

    @torch.no_grad()
    def score_idx(m, idx):
        m.eval()
        idx = np.asarray(idx)
        out = np.empty(len(idx), np.float32)
        step = BATCH * 4
        for i in range(0, len(idx), step):
            b = idx[i:i + step]
            out[i:i + len(b)] = m(torch.from_numpy(get(b)).to(DEVICE)).cpu().numpy()
        return out

    def one_pass(seed):
        """One full CV pass -> out-of-fold scores for all PSMs."""
        oof = np.zeros(n, np.float32)
        for f in range(FOLDS):
            tr_idx = np.where(fold != f)[0]
            te_idx = np.where(fold == f)[0]
            ytr = y[tr_idx]
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
            sample_n = min(len(tr_idx), init_sample_limit)
            if sample_n == len(tr_idx):
                init_idx = tr_idx
            else:
                positions = np.linspace(0, len(tr_idx) - 1, sample_n, dtype=np.int64)
                init_idx = tr_idx[positions]
            Xsamp, ysamp = get(init_idx), y[init_idx]
            # One column at a time, both signs from the SAME column read. This is the
            # same 2*nf q-value evaluations as before (argsort dominates and is kept
            # per sign so tie-ordering is unchanged), but it stops re-slicing the
            # sample matrix twice per feature.
            _t = time.time()
            # Vectorised over feature blocks; see n_targets_at_many. Same counts, same
            # tie-breaking, ~387x fewer Python-level argsort calls.
            best_j, best_sign, best_n = n_targets_at_many(
                Xsamp, ysamp, TRAIN_FDR, topk=env_i("MUMDIA_NN_INIT_TOPK", 0)
            )
            score_tr = (best_sign * get_col(tr_idx, best_j)).astype(np.float32)
            _t = _tick("2_init_feature_scan", _t)
            print(f"  seed {seed} fold {f}: init={feat_cols[best_j]} "
                  f"sign{best_sign:+d} ({best_n}@{TRAIN_FDR:.0%} "
                  f"on {sample_n} training rows)", flush=True)
            model = None
            optim = None
            prev_pos = None
            used_iters = 0
            for _ in range(ITERS):
                q = tda_q(score_tr, ytr)
                pos = (q <= TRAIN_FDR) & (ytr == 1)
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
                        order = np.argsort(-s_neg, kind="stable")
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
                _t = time.time()
                # Warm start reuses the previous iteration's weights and Adam state, so a
                # later iteration only adapts to the changed positive set. WARM_EPOCHS (when
                # set) applies from the SECOND iteration on: the first still needs a full
                # run to get off random initialisation.
                warm_in = (model, optim) if (WARM and model is not None) else None
                ep = None
                if WARM and model is not None and WARM_EPOCHS > 0:
                    ep = WARM_EPOCHS
                model, optim = train_model(sel, pw, seed, warm=warm_in, epochs=ep)
                _t = _tick("3_train", _t)
                score_tr = score_idx(model, tr_idx)
                _t = _tick("4_score_pool_per_iter", _t)
                used_iters += 1
            _t = time.time()
            oof[te_idx] = score_idx(model, te_idx)
            _t = _tick("5_score_holdout", _t)
            print(f"  seed {seed} fold {f}: train targets@{TRAIN_FDR:.0%} = "
                  f"{n_targets_at(score_tr, ytr, TRAIN_FDR)}", flush=True)
        return oof

    # seed ensemble: average rank-normalised out-of-fold scores across seeds
    acc = np.zeros(n, np.float64)
    for s in range(BASE_SEED, BASE_SEED + N_SEEDS):
        np.random.seed(s)
        torch.manual_seed(s)
        oof = one_pass(s)
        acc += pd.Series(oof).rank(method="average").to_numpy() / n
    final = acc / N_SEEDS

    out = pa.table({
        "candidate_id": pa.array(cids.astype(np.uint32), pa.uint32()),
        "score": pa.array(final.astype(np.float64), pa.float64()),
        "q_value": pa.array(np.zeros(n, np.float64), pa.float64()),
    })
    pq.write_table(out, out_path)
    if mm_path and os.path.exists(mm_path):
        try:
            del mm
            os.remove(mm_path)
        except OSError:
            pass
    tot = sum(PHASE.values())
    print("nn_rescore_worker: phase breakdown (wall seconds)", flush=True)
    for k in sorted(PHASE):
        v = PHASE[k]
        print("    %-26s %8.1f s  %5.1f%%" % (k[2:], v, 100 * v / max(tot, 1e-9)), flush=True)
    print("    %-26s %8.1f s" % ("MEASURED TOTAL", tot), flush=True)
    print(f"nn_rescore_worker: {n} PSMs rescored (targets+decoys), {N_SEEDS} seed(s), "
          f"OOF at {FOLDS} folds, backend={'stream' if stream else 'in-memory'}", flush=True)


if __name__ == "__main__":
    main()
