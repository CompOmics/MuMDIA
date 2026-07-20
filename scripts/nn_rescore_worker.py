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
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    stream_env = os.environ.get("MUMDIA_NN_STREAM", "auto").lower()
    filesize = os.path.getsize(pin_path)
    stream = stream_env in ("1", "on", "true") or (
        stream_env == "auto" and filesize > env_f("MUMDIA_NN_STREAM_GB", 4) * 1024 ** 3
    )

    header = pd.read_csv(pin_path, sep="\t", nrows=0).columns.tolist()
    feat_cols = [c for c in header if c not in NON_FEATURE]
    nf = len(feat_cols)
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
        col = lambda j: Xs[:, j]
    else:
        # ---- streaming memmap backend (mean/std, one text pass) ----
        with open(pin_path, "rb") as fh:
            n = sum(1 for _ in fh) - 1
        mm_path = out_path + ".feat.mm"
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
        col = lambda j: np.asarray(mm[:, j])

    # initial direction: best single feature+sign by targets at TRAIN_FDR (on a sample)
    SAMPLE = min(n, env_i("MUMDIA_NN_INIT_SAMPLE", 300000))
    samp = np.arange(SAMPLE)
    Xsamp, ysamp = get(samp), y[samp]
    best_j, best_sign, best_n = 0, 1, -1
    for j in range(nf):
        for sign in (1, -1):
            m_ = n_targets_at(sign * Xsamp[:, j], ysamp, TRAIN_FDR)
            if m_ > best_n:
                best_n, best_j, best_sign = m_, j, sign
    init_score = (best_sign * col(best_j)).astype(np.float32)
    print(f"nn_rescore_worker: device={DEVICE} backend={'stream' if stream else 'in-memory'} "
          f"pool={n} feats={nf} init={feat_cols[best_j]} sign{best_sign:+d} "
          f"({best_n}@{TRAIN_FDR:.0%} on {SAMPLE} sample)", flush=True)

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
        """Minibatch train, drawing each batch's features through `get` (memmap or RAM)."""
        torch.manual_seed(seed)
        m = MLP(nf, HIDDEN, DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
        idx = np.asarray(train_idx)
        for _ in range(EPOCHS):
            m.train()
            perm = idx[np.random.permutation(len(idx))]
            for i in range(0, len(perm), BATCH):
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
            score_tr = init_score[tr_idx].copy()
            model = None
            for _ in range(ITERS):
                q = tda_q(score_tr, ytr)
                pos = (q <= TRAIN_FDR) & (ytr == 1)
                neg = ytr == 0
                sel = tr_idx[pos | neg]
                pw = float(neg.sum()) / max(1.0, float(pos.sum()))
                model = train_model(sel, pw, seed)
                score_tr = score_idx(model, tr_idx)
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
