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
    MUMDIA_NN_STREAM      = auto     auto|1|0  force the streaming memmap backend
    MUMDIA_NN_STREAM_GB   = 4        auto-stream when the PIN exceeds this many GB
    MUMDIA_NN_CHUNK       = 250000   PIN rows per read chunk (streaming backend)
    MUMDIA_NN_INIT_SAMPLE = 300000   rows used to pick the init feature (streaming)
    MUMDIA_NN_EARLY_STOP  = 1        stop self-training once the positive set stabilises
    MUMDIA_NN_EARLY_STOP_TOL = 0.01  churn tolerance for that stop; 0 means exact equality,
                                     which measurably never triggers (dropout + retraining
                                     flips a few borderline PSMs every iteration forever).
                                     Measured on a 40k pool at iters=10: tol 0.01 -> 1.59x
                                     for -0.4% peptides (within NN noise); 0.03 -> 2.15x
                                     for -1.0%; 0.002 -> only 1.06x.
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


def main():
    pin_path, out_path = sys.argv[1], sys.argv[2]

    import torch
    import torch.nn as nn

    FOLDS = env_i("MUMDIA_NN_FOLDS", 3)
    ITERS = env_i("MUMDIA_NN_ITERS", 5)
    EPOCHS = env_i("MUMDIA_NN_EPOCHS", 25)
    HIDDEN = [int(x) for x in os.environ.get("MUMDIA_NN_HIDDEN", "128,64").split(",") if x]
    DROPOUT = env_f("MUMDIA_NN_DROPOUT", 0.3)
    LR = env_f("MUMDIA_NN_LR", 1e-3)
    WD = env_f("MUMDIA_NN_WD", 1e-4)
    BATCH = env_i("MUMDIA_NN_BATCH", 4096)
    TRAIN_FDR = env_f("MUMDIA_NN_TRAIN_FDR", 0.01)
    N_SEEDS = env_i("MUMDIA_NN_SEEDS", 1)
    CHUNK = env_i("MUMDIA_NN_CHUNK", 250000)
    EARLY_STOP = env_i("MUMDIA_NN_EARLY_STOP", 1) != 0
    EARLY_STOP_TOL = env_f("MUMDIA_NN_EARLY_STOP_TOL", 0.01)
    PREGATHER_GB = env_f("MUMDIA_NN_PREGATHER_GB", 8)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if FOLDS < 2 or ITERS < 1 or EPOCHS < 1 or N_SEEDS < 1:
        raise ValueError("folds>=2, iterations>=1, epochs>=1, and seeds>=1 are required")

    # Torch CPU threads. Measured: the tiny MLP saturates at ~16 threads and is SLOWER at
    # 32 (oversubscription on small GEMMs), so cap rather than inherit all cores. An
    # explicit OMP_NUM_THREADS from the caller wins.
    if DEVICE == "cpu":
        want = env_i("MUMDIA_NN_THREADS", 16)
        if "OMP_NUM_THREADS" in os.environ:
            want = env_i("OMP_NUM_THREADS", want)
        if want > 0:
            torch.set_num_threads(max(1, min(want, os.cpu_count() or want)))

    stream_env = os.environ.get("MUMDIA_NN_STREAM", "auto").lower()
    filesize = os.path.getsize(pin_path)
    stream_gb = env_f("MUMDIA_NN_STREAM_GB", 4)
    stream = stream_env in ("1", "on", "true") or (
        stream_env == "auto" and filesize > stream_gb * 1024 ** 3
    )
    # Say WHY a backend was chosen. The auto-threshold is on PIN *text* size, so crossing
    # ~1M PSMs silently switched to the disk-backed memmap and started requiring a
    # writable path -- an invisible change of behaviour when it went wrong.
    print(
        f"nn_rescore_worker: PIN {filesize / 1024 ** 3:.2f} GB, threshold {stream_gb:.2f} GB, "
        f"MUMDIA_NN_STREAM={stream_env} -> backend={'stream(memmap)' if stream else 'in-memory'}",
        flush=True,
    )

    header = pd.read_csv(pin_path, sep="\t", nrows=0).columns.tolist()
    feat_cols = [c for c in header if c not in NON_FEATURE]
    nf = len(feat_cols)
    if nf == 0:
        raise ValueError("PIN contains no rescoring feature columns")
    fold_of = lambda p: int(hashlib.md5(strip_pep(p).encode()).hexdigest(), 16) % FOLDS

    mm_path = None
    if not stream:
        # ---- in-memory backend (median/IQR standardisation) ----
        pin = pd.read_csv(pin_path, sep="\t")
        y = (pin["Label"].to_numpy() == 1).astype(np.float32)
        cids = np.array([int(s.rsplit("_", 1)[-1]) for s in pin["SpecId"].astype(str)], np.int64)
        fold = pin["Peptide"].map(fold_of).to_numpy()
        X = np.nan_to_num(pin[feat_cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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
        for chunk in pd.read_csv(pin_path, sep="\t", usecols=lambda c: c in keep, chunksize=CHUNK):
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

    def train_model(train_idx, pos_weight, seed):
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
        m = MLP(nf, HIDDEN, DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
        idx = np.asarray(train_idx)
        ntr = len(idx)
        pregather = (ntr * nf * 4) <= PREGATHER_GB * 1024 ** 3
        if pregather:
            Xt = torch.from_numpy(np.ascontiguousarray(get(idx))).to(DEVICE)
            yt = torch.from_numpy(np.ascontiguousarray(y[idx])).to(DEVICE)
        for _ in range(EPOCHS):
            m.train()
            # numpy RNG drives the shuffle in both paths, so the training trajectory
            # stays tied to the existing np.random.seed(seed) stream.
            order = np.random.permutation(ntr)
            if pregather:
                perm_t = torch.from_numpy(order).to(DEVICE)
                for i in range(0, ntr, BATCH):
                    b = perm_t[i:i + BATCH]
                    opt.zero_grad()
                    lossf(m(Xt[b]), yt[b]).backward()
                    opt.step()
            else:
                perm = idx[order]
                for i in range(0, ntr, BATCH):
                    b = perm[i:i + BATCH]
                    Xb = torch.from_numpy(get(b)).to(DEVICE)
                    yb = torch.from_numpy(y[b]).to(DEVICE)
                    opt.zero_grad()
                    lossf(m(Xb), yb).backward()
                    opt.step()
        return m

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
            best_j, best_sign, best_n = 0, 1, -1
            for j in range(nf):
                col = np.ascontiguousarray(Xsamp[:, j])
                for sign in (1, -1):
                    count = n_targets_at(sign * col, ysamp, TRAIN_FDR)
                    if count > best_n:
                        best_n, best_j, best_sign = count, j, sign
            score_tr = (best_sign * get_col(tr_idx, best_j)).astype(np.float32)
            print(f"  seed {seed} fold {f}: init={feat_cols[best_j]} "
                  f"sign{best_sign:+d} ({best_n}@{TRAIN_FDR:.0%} "
                  f"on {sample_n} training rows)", flush=True)
            model = None
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
                pw = float(neg.sum()) / max(1.0, float(pos.sum()))
                model = train_model(sel, pw, seed)
                score_tr = score_idx(model, tr_idx)
                used_iters += 1
            oof[te_idx] = score_idx(model, te_idx)
            print(f"  seed {seed} fold {f}: train targets@{TRAIN_FDR:.0%} = "
                  f"{n_targets_at(score_tr, ytr, TRAIN_FDR)}", flush=True)
        return oof

    # seed ensemble: average rank-normalised out-of-fold scores across seeds
    acc = np.zeros(n, np.float64)
    for s in range(N_SEEDS):
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
    print(f"nn_rescore_worker: {n} PSMs rescored (targets+decoys), {N_SEEDS} seed(s), "
          f"OOF at {FOLDS} folds, backend={'stream' if stream else 'in-memory'}", flush=True)


if __name__ == "__main__":
    main()
