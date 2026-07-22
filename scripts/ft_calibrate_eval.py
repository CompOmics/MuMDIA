"""DeepLC CALIBRATION (no fine-tuning) vs fine-tune vs DIA-NN, on a common random
held-out split. Calibration fits a calibration curve + selects the best internal MT
model on the reference set WITHOUT changing weights, so it cannot overfit like
fine-tuning. Reports held-out MAD for:
  (a) DeepLC native predict_and_calibrate (calibrated seconds, its own curve),
  (b) DeepLC base predict + binned-median calibration (isolates base-model quality,
      same calibrator used for DIA-NN),
  (c) DIA-NN raw library iRT + binned-median calibration (baseline).

Usage: python ft_calibrate_eval.py <seed> --q-train Q --held-frac F --seed S --raw-lib LIB
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
is_std = lambda pf: all(c in STD for c in re.sub(r"\[[^\]]*\]", "", pf))


def calibrate_fit(x, y, nbins=80):
    o = np.argsort(x); xs, ys = x[o], y[o]
    e = np.linspace(0, len(xs), nbins + 1).astype(int)
    cx, cy = [], []
    for b in range(nbins):
        lo, hi = e[b], e[b + 1]
        if hi > lo:
            cx.append(np.median(xs[lo:hi])); cy.append(np.median(ys[lo:hi]))
    cx, cy = np.array(cx), np.array(cy)
    return lambda q: np.interp(q, cx, cy)


def mad(pred, obs):
    r = np.asarray(pred, float) - np.asarray(obs, float)
    return np.median(np.abs(r - np.median(r)))


def flat(a):
    a = np.asarray(a, float)
    return a.mean(axis=1) if a.ndim == 2 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_path")
    ap.add_argument("--q-train", dest="q_train", type=float, default=0.01)
    ap.add_argument("--held-frac", dest="held_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(_T))
    ap.add_argument("--raw-lib", dest="raw_lib", default=None)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    s = pq.read_table(a.seed_path).to_pydict()
    have_irt = "predicted_irt" in s
    best = {}
    for i in range(len(s["peptidoform"])):
        if s["label"][i] != "target":
            continue
        pf = s["peptidoform"][i]
        if not is_std(pf):
            continue
        b = s["base_peptide_id"][i]
        if b not in best or s["score"][i] > best[b][0]:
            irt = float(s["predicted_irt"][i]) if have_irt else float("nan")
            best[b] = (s["score"][i], pf, s["observed_rt"][i], s["spectrum_q"][i], irt)
    conf = [(pf, rt, q) for (_, pf, rt, q, _) in best.values() if q < 0.01]
    # raw iRT per peptidoform straight from the seed (the library iRT that was used),
    # so the DIA-NN baseline needs no external lib file.
    seed_irt = {pf: irt for (_, pf, rt, q, irt) in best.values()}
    rng = np.random.default_rng(a.seed)
    mask = rng.random(len(conf)) < a.held_frac
    held = [(pf, rt) for i, (pf, rt, q) in enumerate(conf) if mask[i]]
    train = [(pf, rt) for i, (pf, rt, q) in enumerate(conf) if not mask[i] and q < a.q_train]
    print(f"q_train={a.q_train} train={len(train)} held={len(held)}", flush=True)

    ref = PSMList(psm_list=[PSM(peptidoform=pf, retention_time=rt, spectrum_id=str(k))
                            for k, (pf, rt) in enumerate(train)])
    tr_pf = [pf for pf, _ in train]; tr_obs = np.array([rt for _, rt in train], float)
    hd_pf = [pf for pf, _ in held]; hd_obs = np.array([rt for _, rt in held], float)

    # (a) DeepLC native predict_and_calibrate (calibrated seconds)
    tr_cal = flat(deeplc.predict_and_calibrate(tr_pf, psm_list_reference=ref))
    hd_cal = flat(deeplc.predict_and_calibrate(hd_pf, psm_list_reference=ref))
    print(f"CAL   q_train={a.q_train} train_MAD={mad(tr_cal, tr_obs):.3f} held_MAD={mad(hd_cal, hd_obs):.3f}", flush=True)

    # (b) DeepLC base predict + binned-median calibration
    tr_base = flat(deeplc.predict(tr_pf)); hd_base = flat(deeplc.predict(hd_pf))
    cb = calibrate_fit(tr_base, tr_obs)
    print(f"BASE  q_train={a.q_train} train_MAD={mad(cb(tr_base), tr_obs):.3f} held_MAD={mad(cb(hd_base), hd_obs):.3f}", flush=True)

    # (c) DIA-NN raw. Prefer an external lib; else use the seed's own predicted_irt.
    rm = None
    if a.raw_lib:
        rl = pq.read_table(a.raw_lib, columns=["peptidoform", "predicted_irt"]).to_pydict()
        rm = {}
        for pf, irt in zip(rl["peptidoform"], rl["predicted_irt"]):
            rm.setdefault(pf, float(irt))
    elif have_irt:
        rm = {pf: irt for pf, irt in seed_irt.items() if np.isfinite(irt)}
    if rm:
        rtr = np.array([rm[pf] for pf in tr_pf if pf in rm], float)
        rtro = np.array([rt for pf, rt in train if pf in rm], float)
        rhd = np.array([rm[pf] for pf in hd_pf if pf in rm], float)
        rhdo = np.array([rt for pf, rt in held if pf in rm], float)
        cr = calibrate_fit(rtr, rtro)
        print(f"DIANN q_train={a.q_train} train_MAD={mad(cr(rtr), rtro):.3f} held_MAD={mad(cr(rhd), rhdo):.3f}", flush=True)


if __name__ == "__main__":
    main()
