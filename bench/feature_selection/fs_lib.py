"""Shared pieces of the feature-selection study (docs/28).

Loads a PIN (parquet from `pin2parquet.py`, or the tab-separated original), computes
target-decoy q-values, and runs a faithful re-implementation of `scripts/nn_rescore_worker.py`
(same folds, init-feature scan, self-training loop, MLP, early stop) on any column subset,
so that a feature subset is judged on the quantity the engine optimises: stripped peptides at
1% FDR, with the decoy fraction as the validity check.

Analysis only: nothing here is imported by the engine or the worker.
"""
from __future__ import annotations

import hashlib
import os
import re
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NON_FEATURE = {"SpecId", "Label", "ScanNr", "ExpMass", "CalcMass", "Peptide", "Proteins"}

# The configuration the engine ships: `RescoreConfig` defaults (folds 3, num_iter 10,
# train_fdr 0.01) are passed to the worker as MUMDIA_NN_FOLDS / ITERS / TRAIN_FDR
# (rescore.rs), overriding the worker's own docstring default of 5 iterations. The measured
# HYE run trained 10 iterations per fold and never hit the churn stop.
WORKER_DEFAULTS = dict(
    folds=3,
    iters=10,
    epochs=25,
    hidden=(128, 64),
    dropout=0.3,
    lr=1e-3,
    wd=1e-4,
    batch=4096,
    train_fdr=0.01,
    early_stop=True,
    early_stop_tol=0.01,
    init_sample=300_000,
    init_topk=0,
    seeds=1,
    seed_base=0,
    device="cuda",
    score_step=4096 * 4,
)


def strip_pep(p: str) -> str:
    s = re.sub(r"\[[^\]]*\]", "", str(p))
    s = re.sub(r"^[A-Z-]\.", "", s)
    s = re.sub(r"\.[A-Z-]$", "", s)
    return s


def load_pin(path: str, chunk: int = 250_000, standardise: bool = False) -> dict:
    """Return X (float32, NaN/inf -> 0), y (1 = target), fold hash, stripped sequence, names.

    `standardise=False` keeps the raw values for the structural statistics; the objective
    runs standardise per subset themselves (mean/std, clipped to +-8, as the worker's
    streaming backend does; the in-memory backend uses median/IQR, which only rescales).
    """
    t0 = time.time()
    header = list(pq.read_schema(path).names)
    feat_cols = [c for c in header if c not in NON_FEATURE]
    tb = pq.read_table(path, columns=["SpecId", "Label", "Peptide"])
    y = (tb.column("Label").to_numpy() == 1).astype(np.float32)
    cids = np.array([int(x.rsplit("_", 1)[-1]) for x in tb.column("SpecId").to_pylist()], np.int64)
    peps = [strip_pep(p) for p in tb.column("Peptide").to_pylist()]
    pep_hash = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) for p in peps], dtype=object)
    base_seq = np.array([p[6:] if p.startswith("DECOY_") else p for p in peps], dtype=object)
    del tb
    n, nf = len(y), len(feat_cols)
    X = np.empty((n, nf), np.float32)
    pf = pq.ParquetFile(path)
    off = 0
    for b in pf.iter_batches(batch_size=chunk, columns=feat_cols):
        k = b.num_rows
        for j in range(nf):
            X[off : off + k, j] = b.column(j).to_numpy(zero_copy_only=False)
        off += k
    nan_frac = np.isnan(X).mean(axis=0)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(
        f"loaded {n} PSMs x {nf} features in {time.time() - t0:.0f}s; "
        f"targets {int(y.sum())}, decoys {int(n - y.sum())}",
        flush=True,
    )
    d = dict(X=X, y=y, cids=cids, pep_hash=pep_hash, base_seq=base_seq, feat_cols=feat_cols, nan_frac=nan_frac)
    if standardise:
        d["X"] = standardise_inplace(d["X"])
    return d


def standardise_inplace(X: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    n = X.shape[0]
    mean = X.mean(axis=0, dtype=np.float64)
    var = (X.astype(np.float64) ** 2).mean(axis=0) - mean**2 if n < 2_000_000 else None
    if var is None:
        s2 = np.zeros(X.shape[1], np.float64)
        for i in range(0, n, chunk):
            blk = X[i : i + chunk].astype(np.float64)
            s2 += (blk**2).sum(axis=0)
        var = s2 / n - mean**2
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    std[std == 0] = 1.0
    mean = mean.astype(np.float32)
    for i in range(0, n, chunk):
        X[i : i + chunk] = np.clip((X[i : i + chunk] - mean) / std, -8, 8)
    return X


def subsample(d: dict, n_rows: int | None, seed: int = 0) -> dict:
    if n_rows is None or n_rows >= len(d["y"]):
        return d
    rs = np.random.RandomState(seed)
    idx = np.sort(rs.choice(len(d["y"]), size=n_rows, replace=False))
    n = len(d["y"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in d.items()}


# ----------------------------------------------------------------------------- FDR


def tda_q(scores: np.ndarray, is_target: np.ndarray) -> np.ndarray:
    """Target-decoy q-values: FDR = (decoys + 1) / max(1, targets) down the ranking, q = running min."""
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
    ct = np.cumsum(t_sorted, dtype=np.int64)
    cd = np.cumsum(~t_sorted, dtype=np.int64)
    ok = (cd + 1) <= fdr * np.maximum(ct, 1)
    if not ok.any():
        return 0, -1
    last = int(np.flatnonzero(ok)[-1])
    return int(ct[last]), last


def n_targets_at_col(col, tgt, fdr, topk=0):
    n = col.shape[0]
    if topk and topk < n:
        idx = np.argpartition(-col, topk - 1)[:topk]
        idx = idx[np.argsort(-col[idx], kind="stable")]
        cnt, last = _count_at_fdr_sorted(tgt[idx], fdr)
        if last < topk - 1:
            return cnt
    order = np.argsort(-col, kind="stable")
    return _count_at_fdr_sorted(tgt[order], fdr)[0]


def n_targets_at_many(X, is_target, fdr, topk=0):
    tgt = np.asarray(is_target).astype(bool)
    best_j, best_sign, best_n = 0, 1, -1
    for j in range(X.shape[1]):
        col = np.ascontiguousarray(X[:, j])
        for sign in (1, -1):
            c = n_targets_at_col(col if sign > 0 else -col, tgt, fdr, topk=topk)
            if c > best_n:
                best_n, best_j, best_sign = c, j, sign
    return best_j, best_sign, best_n


def evaluate(score: np.ndarray, d: dict, fdr: float = 0.01) -> dict:
    """PSMs and picked stripped peptides at `fdr`, plus the decoy fraction among accepted PSMs."""
    y = d["y"]
    q = tda_q(score.astype(np.float64), y)
    acc = q <= fdr
    n_psm = int((acc & (y == 1)).sum())
    n_dec = int((acc & (y == 0)).sum())
    df = pd.DataFrame({"seq": d["base_seq"], "score": score, "y": y})
    best = df.sort_values("score", ascending=False).drop_duplicates("seq")
    qp = tda_q(best["score"].to_numpy(np.float64), best["y"].to_numpy())
    n_pep = int(((qp <= fdr) & (best["y"].to_numpy() == 1)).sum())
    return dict(psms_1pct=n_psm, decoys_1pct=n_dec, decoy_frac=round(n_dec / max(1, n_psm), 4), peptides_1pct=n_pep)


# ----------------------------------------------------------------------------- the worker, re-implemented


def run_rescoring(d: dict, cols: list[int] | None = None, cfg: dict | None = None, verbose: bool = False) -> dict:
    """Out-of-fold NN scores for every PSM of `d`, using only the feature columns `cols`.

    Mirrors scripts/nn_rescore_worker.py: md5(stripped peptide) folds, per-fold init feature
    scan over both signs, Percolator-style self-training on targets at `train_fdr` vs all
    decoys, a fresh MLP per iteration, churn-based early stop, rank-averaged seeds.
    """
    import torch
    import torch.nn as nn

    cfg = {**WORKER_DEFAULTS, **(cfg or {})}
    dev = cfg["device"] if torch.cuda.is_available() else "cpu"
    Xall, y = d["X"], d["y"]
    if cols is None:
        cols = list(range(Xall.shape[1]))
    X = np.ascontiguousarray(Xall[:, cols])
    X = standardise_inplace(X.copy())
    n, nf = X.shape
    fold = (d["pep_hash"] % cfg["folds"]).astype(np.int16)
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    phase: dict[str, float] = {}

    class MLP(nn.Module):
        def __init__(self, d_in, hidden, p):
            super().__init__()
            layers, dd = [], d_in
            for h in hidden:
                layers += [nn.Linear(dd, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
                dd = h
            layers += [nn.Linear(dd, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_model(train_idx, pos_weight, seed):
        torch.manual_seed(seed)
        m = MLP(nf, list(cfg["hidden"]), cfg["dropout"]).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
        idx_t = torch.as_tensor(np.asarray(train_idx), device=dev)
        Xb, yb = Xt[idx_t], yt[idx_t]
        ntr, B = len(idx_t), cfg["batch"]
        for _ in range(cfg["epochs"]):
            m.train()
            order = torch.from_numpy(np.random.permutation(ntr)).to(dev)
            n_use = ntr - 1 if (ntr % B) == 1 and ntr > B else ntr
            for i in range(0, n_use, B):
                b = order[i : i + B]
                opt.zero_grad(set_to_none=True)
                loss = lossf(m(Xb[b]).float(), yb[b])
                loss.backward()
                opt.step()
        return m

    @torch.no_grad()
    def score_idx(m, idx):
        m.eval()
        idx_t = torch.as_tensor(np.asarray(idx), device=dev)
        out = torch.empty(len(idx_t), dtype=torch.float32, device=dev)
        step = cfg["score_step"]
        for i in range(0, len(idx_t), step):
            b = idx_t[i : i + step]
            out[i : i + len(b)] = m(Xt[b]).float()
        return out.cpu().numpy()

    def tick(name, t0):
        phase[name] = phase.get(name, 0.0) + (time.time() - t0)
        return time.time()

    iters_used = []
    inits = []
    t_start = time.time()

    def one_pass(seed):
        oof = np.zeros(n, np.float32)
        for f in range(cfg["folds"]):
            tr_idx = np.where(fold != f)[0]
            te_idx = np.where(fold == f)[0]
            ytr = y[tr_idx]
            t = time.time()
            sample_n = min(len(tr_idx), cfg["init_sample"])
            init_idx = tr_idx if sample_n == len(tr_idx) else tr_idx[np.linspace(0, len(tr_idx) - 1, sample_n, dtype=np.int64)]
            best_j, best_sign, best_n = n_targets_at_many(X[init_idx], y[init_idx], cfg["train_fdr"], topk=cfg["init_topk"])
            inits.append((d["feat_cols"][cols[best_j]], best_sign, best_n))
            score_tr = (best_sign * X[tr_idx, best_j]).astype(np.float32)
            t = tick("init", t)
            model = prev_pos = None
            used = 0
            for _ in range(cfg["iters"]):
                t = time.time()
                q = tda_q(score_tr, ytr)
                pos = (q <= cfg["train_fdr"]) & (ytr == 1)
                neg = ytr == 0
                if model is not None and prev_pos is not None:
                    churn = int(np.count_nonzero(pos != prev_pos))
                    frac = churn / max(1, int(pos.sum()))
                    if cfg["early_stop"] and frac <= cfg["early_stop_tol"]:
                        break
                prev_pos = pos
                sel = tr_idx[pos | neg]
                pw = float(neg.sum()) / max(1.0, float(pos.sum()))
                model = train_model(sel, pw, seed)
                if dev != "cpu":
                    torch.cuda.synchronize()
                t = tick("train", t)
                score_tr = score_idx(model, tr_idx)
                t = tick("score", t)
                used += 1
            iters_used.append(used)
            oof[te_idx] = score_idx(model, te_idx)
            if verbose:
                print(f"  seed {seed} fold {f}: {used} iters, train targets@1% = {n_targets_at(score_tr, ytr, cfg['train_fdr'])}", flush=True)
        return oof

    acc = np.zeros(n, np.float64)
    for s in range(cfg["seeds"]):
        sd = s + cfg["seed_base"]
        np.random.seed(sd)
        torch.manual_seed(sd)
        acc += pd.Series(one_pass(sd)).rank(method="average").to_numpy() / n
    final = acc / cfg["seeds"]
    wall = time.time() - t_start
    del Xt, yt
    if dev != "cpu":
        torch.cuda.empty_cache()
    return dict(score=final, phase=phase, wall=wall, iters_used=iters_used, inits=inits, n_features=nf)


def bench_subset(name: str, d: dict, cols: list[int] | None, cfg: dict | None = None, verbose: bool = False) -> dict:
    r = run_rescoring(d, cols, cfg, verbose=verbose)
    m = evaluate(r["score"], d)
    row = dict(
        variant=name,
        n_features=r["n_features"],
        **m,
        wall_s=round(r["wall"], 1),
        train_s=round(r["phase"].get("train", 0.0), 1),
        iters=sum(r["iters_used"]),
        init=";".join(sorted({i[0] for i in r["inits"]})),
    )
    print(row, flush=True)
    return row, r


# ----------------------------------------------------------------------------- feature families

FAMILY_ORDER = [
    "similarity", "entropy", "coelution", "interference", "chromatographic", "mass_accuracy",
    "ion_series", "ms1", "rt", "novel", "nonzero", "order_consistency", "peak_scans",
    "apex_dispersion", "mass_uncertainty", "demix",
]


def family_map(repo_root: str) -> dict[str, str]:
    """Feature name -> family, read from the Rust registry so the report cannot drift from it.

    `minimal` and `rich` are the legacy sets (features.rs MINIMAL_FEATURES / RICH_EXTRA); the
    extended families come from `stages/features/<family>.rs` NAMES in registry order, first
    family wins for a repeated name (which is what `extended_name_refs` does); the six
    psms-derived extras are `psm_extra`.
    """
    import re as _re

    feats = os.path.join(repo_root, "rust", "mumdia", "crates", "mumdia", "src", "stages", "features.rs")
    src = open(feats, encoding="utf-8").read()

    def const_list(name):
        m = _re.search(r"pub const %s: &\[&str\] = &\[(.*?)\];" % name, src, _re.S)
        return _re.findall(r'"([A-Za-z0-9_]+)"', m.group(1))

    out: dict[str, str] = {}
    for n in const_list("MINIMAL_FEATURES"):
        out.setdefault(n, "minimal")
    for n in const_list("RICH_EXTRA"):
        out.setdefault(n, "rich")
    fam_dir = os.path.join(os.path.dirname(feats), "features")
    for fam in FAMILY_ORDER:
        s = open(os.path.join(fam_dir, fam + ".rs"), encoding="utf-8").read()
        m = _re.search(r"pub const NAMES: &\[&str\] = &\[(.*?)\];", s, _re.S)
        for n in _re.findall(r'"([A-Za-z0-9_]+)"', m.group(1)):
            out.setdefault(n, fam)
    for n in [
        "peak_contested_frac", "peak_contested_count_frac", "peak_apportioned_frac",
        "n_charge_states", "charge_multi_flag", "cross_charge_intensity_log",
    ]:
        out.setdefault(n, "psm_extra")
    return out
