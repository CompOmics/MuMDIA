"""Fast DeepLC fine-tune epoch sweep for generalization. Fine-tune on the q<q_train
confident seed peptides, predict ONLY the confident peptides (train + held-out),
LOESS-calibrate on train, and report train vs held-out RT residual MAD. Held-out =
q in [q_train, 0.01], unseen by fine-tune -> detects overfitting. Skips the full
3.74M-peptide library prediction so each epoch setting costs only its training time.

Usage: python ft_epoch_eval.py <seed> <epochs> [--q-train Q] [--patience P]
"""
import os
_T = os.environ.get("DEEPLC_FT_THREADS", "8")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

import argparse
import re
import deeplc
import numpy as np
import pyarrow.parquet as pq
import torch
from psm_utils import PSM, PSMList

STD = set("ACDEFGHIKLMNPQRSTVWY")
strip_mods = lambda s: re.sub(r"\[[^\]]*\]", "", s)
is_std = lambda pf: all(c in STD for c in strip_mods(pf))


def calibrate_fit(x, y, nbins=80):
    """Robust monotone calibration curve (quantile-binned medians), numpy-only
    stand-in for LOESS. Returns a predictor callable via np.interp."""
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    edges = np.linspace(0, len(xs), nbins + 1).astype(int)
    cx, cy = [], []
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            cx.append(np.median(xs[lo:hi]))
            cy.append(np.median(ys[lo:hi]))
    cx, cy = np.array(cx), np.array(cy)
    return lambda q: np.interp(q, cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_path")
    ap.add_argument("epochs", type=int)
    ap.add_argument("--q-train", dest="q_train", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512,
                    help="fine-tune batch size; small sets need a small batch so each "
                         "epoch has enough gradient steps to converge")
    ap.add_argument("--threads", type=int, default=int(_T))
    ap.add_argument("--held-frac", dest="held_frac", type=float, default=0.0,
                    help="if >0, hold out this random fraction of the confident (q<0.01) "
                         "set (excluded from training regardless of q_train); the rest, "
                         "subject to q_train, is the train set. Enables fair comparison "
                         "of q_train choices on a common held-out set. 0 = legacy q-band held-out.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the random held-out split")
    ap.add_argument("--raw-lib", dest="raw_lib", default=None,
                    help="optional library parquet; also report the raw (un-fine-tuned) "
                         "iRT held-out MAD on the SAME split, as the baseline to beat")
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    s = pq.read_table(a.seed_path).to_pydict()
    # best-scoring target PSM per base_peptide_id
    best = {}
    for i in range(len(s["peptidoform"])):
        if s["label"][i] != "target":
            continue
        pf = s["peptidoform"][i]
        if not is_std(pf):
            continue
        b = s["base_peptide_id"][i]
        if b not in best or s["score"][i] > best[b][0]:
            best[b] = (s["score"][i], pf, s["observed_rt"][i], s["spectrum_q"][i])
    rows = list(best.values())
    conf = [(pf, rt, q) for (_, pf, rt, q) in rows if q < 0.01]  # confident universe
    if a.held_frac > 0:
        # Random split of the confident set: `held` is fixed across q_train choices so
        # different train-set sizes are compared on the SAME held-out peptides.
        rng = np.random.default_rng(a.seed)
        mask = rng.random(len(conf)) < a.held_frac
        held = [(pf, rt) for i, (pf, rt, q) in enumerate(conf) if mask[i]]
        train = [(pf, rt) for i, (pf, rt, q) in enumerate(conf) if not mask[i] and q < a.q_train]
    else:
        train = [(pf, rt) for (pf, rt, q) in conf if q < a.q_train]
        held = [(pf, rt) for (pf, rt, q) in conf if a.q_train <= q < 0.01]
    print(f"epochs={a.epochs} q_train={a.q_train} held_frac={a.held_frac}  "
          f"train={len(train)} held={len(held)}", flush=True)

    ref = PSMList(psm_list=[PSM(peptidoform=pf, retention_time=rt, spectrum_id=str(k))
                            for k, (pf, rt) in enumerate(train)])
    tk = {"num_workers": 0, "epochs": a.epochs, "batch_size": a.batch,
          "patience": a.patience, "device": "cpu", "num_threads": max(1, a.threads)}
    model = deeplc.finetune(ref, train_kwargs=tk)

    def predict(items):
        pf = [p for p, _ in items]
        pr = deeplc.predict(pf, model=model)
        pr = np.asarray(pr, float)
        if pr.ndim == 2:
            pr = pr.mean(axis=1)
        return pr, np.array([rt for _, rt in items], float)

    tr_irt, tr_obs = predict(train)
    hd_irt, hd_obs = predict(held)
    cal = calibrate_fit(tr_irt, tr_obs)

    def mad(pred, obs):
        r = pred - obs
        return np.median(np.abs(r - np.median(r)))

    tr_mad = mad(cal(tr_irt), tr_obs)
    hd_mad = mad(cal(hd_irt), hd_obs)
    print(f"RESULT q_train={a.q_train} epochs={a.epochs} train_MAD={tr_mad:.3f} held_MAD={hd_mad:.3f}", flush=True)

    # Raw (un-fine-tuned) baseline on the identical train/held split.
    if a.raw_lib:
        rl = pq.read_table(a.raw_lib, columns=["peptidoform", "predicted_irt"]).to_pydict()
        rawmap = {}
        for pf, irt in zip(rl["peptidoform"], rl["predicted_irt"]):
            rawmap.setdefault(pf, float(irt))
        rtr = np.array([rawmap[pf] for pf, _ in train if pf in rawmap], float)
        rtr_o = np.array([rt for pf, rt in train if pf in rawmap], float)
        rhd = np.array([rawmap[pf] for pf, _ in held if pf in rawmap], float)
        rhd_o = np.array([rt for pf, rt in held if pf in rawmap], float)
        rcal = calibrate_fit(rtr, rtr_o)
        print(f"RAW q_train={a.q_train} train_MAD={mad(rcal(rtr), rtr_o):.3f} "
              f"held_MAD={mad(rcal(rhd), rhd_o):.3f}", flush=True)


if __name__ == "__main__":
    main()
