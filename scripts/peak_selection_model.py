"""Peak-selection model (sensitivity_plan backlog P1.3, spec 03 §6 peak-selection).

First-pass, grouped out-of-fold peak scoring over the top-K peak table emitted by
`extract` when `retain_top_peaks > 1` (`<psms>.peaks.parquet`). The question is:
given several retained chromatographic peaks for one precursor, which one is the
right one? This trains a simple ranker on per-peak descriptors and reports how
often it puts the correct peak at rank 1 / in the top 3, versus the raw
evidence-count and area rankings.

Correctness label:
  - with --diann (a DIA-NN report giving a reference apex RT per precursor): the
    correct peak is the retained peak whose [start_rt, end_rt] contains, or whose
    apex is nearest (within --rt-tol-s) to, the reference apex. This is the honest
    label.
  - without --diann: a WEAK self-label is used (the peak nearest the apex the
    engine currently selected, from psms.apex_rt). This only measures whether the
    ranker reproduces the current heuristic, not correctness, and is printed with
    that caveat.

Grouping: all peaks of one candidate stay in the same CV fold (group = candidate).
Leakage guards: fit scaling inside the training fold; the label is never a feature.

Caveat (P1.2 dependency): the peak table carries apex/boundaries/evidence/area but
not a full per-peak feature vector (the engine computes features only for the
selected apex). A production peak-selection model needs per-peak features; this
first pass uses the peak-shape descriptors that are available.

Usage:
  python peak_selection_model.py --peaks <psms>.peaks.parquet --psms psms.parquet
         [--diann report] [--folds 3] [--rt-tol-s 10] [--out metrics.json]
Requires an env with scikit-learn + pyarrow (py312_mumdia). Deterministic.
"""
import argparse
import hashlib
import json
import re
import sys

import numpy as np
import pyarrow.parquet as pq

strip = lambda p: re.sub(r"\[[^\]]*\]", "", str(p))


def fold_of(cid, folds):
    h = hashlib.sha1(str(int(cid)).encode()).hexdigest()
    return int(h, 16) % folds


def load_diann(path):
    """Return dict (stripped_seq, charge) -> apex RT in seconds. Defensive columns."""
    t = pq.read_table(path).to_pandas() if path.endswith(".parquet") else None
    if t is None:
        import pandas as pd
        sep = "\t" if path.endswith((".tsv", ".txt")) else ","
        t = pd.read_csv(path, sep=sep)

    def pick(*cands):
        for c in cands:
            if c in t.columns:
                return c
        return None

    seqc = pick("Modified.Sequence", "ModifiedPeptide", "Stripped.Sequence", "Peptide")
    zc = pick("Precursor.Charge", "Charge", "PrecursorCharge")
    rtc = pick("RT", "iRT", "Retention.Time", "RT.Start", "Apex.RT")
    if not (seqc and zc and rtc):
        raise SystemExit(f"peak_selection_model: DIA-NN columns not found in {path}")
    rt = t[rtc].to_numpy(dtype=float)
    # detect minutes vs seconds by magnitude (gradients are usually > 20 min in s)
    if np.nanmax(rt) < 300:
        rt = rt * 60.0
    out = {}
    for s, z, r in zip(t[seqc].astype(str), t[zc].astype(int), rt):
        out[(strip(s), int(z))] = float(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--psms", required=True)
    ap.add_argument("--diann", default=None)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--rt-tol-s", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    pk = pq.read_table(args.peaks).to_pandas()
    ps = pq.read_table(args.psms, columns=["candidate_id", "apex_rt", "peptidoform", "charge", "label"]).to_pandas()
    sel_rt = dict(zip(ps.candidate_id.astype(int), ps.apex_rt.astype(float)))
    key = {int(c): (strip(p), int(z)) for c, p, z in zip(ps.candidate_id, ps.peptidoform, ps.charge)}

    diann = load_diann(args.diann) if args.diann else None
    label_kind = "diann-reference" if diann else "weak-self (current apex)"

    # Per-candidate peak groups.
    pk = pk.sort_values(["candidate_id", "peak_rank"]).reset_index(drop=True)
    feats, labels, groups = [], [], []
    per_cand = {}
    n_labeled = 0
    for cid, g in pk.groupby("candidate_id"):
        cid = int(cid)
        ev = g.evidence_count.to_numpy(float)
        ar = g.area.to_numpy(float)
        ap = g.apex_rt.to_numpy(float)
        width = (g.end_rt.to_numpy(float) - g.start_rt.to_numpy(float))
        # reference RT for the label
        ref = None
        if diann is not None:
            ref = diann.get(key.get(cid))
        else:
            ref = sel_rt.get(cid)
        if ref is None:
            continue
        # correct peak = nearest apex within tol (or containing the ref)
        d = np.abs(ap - ref)
        if d.min() > args.rt_tol_s:
            continue  # no retained peak matches the reference -> unlabelable here
        correct = int(np.argmin(d))
        n_labeled += 1
        mx_ev = ev.max() if ev.max() > 0 else 1.0
        mx_ar = ar.max() if ar.max() > 0 else 1.0
        for j in range(len(g)):
            feats.append([ev[j], ar[j], float(g.peak_rank.iloc[j]), width[j], ev[j] / mx_ev, ar[j] / mx_ar])
            labels.append(1 if j == correct else 0)
            groups.append(cid)
        per_cand[cid] = (ev, ar, correct)

    if n_labeled < 20:
        raise SystemExit(f"peak_selection_model: too few labelable candidates ({n_labeled})")
    X = np.asarray(feats, float)
    y = np.asarray(labels, int)
    grp = np.asarray(groups, int)

    # Out-of-fold scores.
    oof = np.zeros(len(y))
    folds = np.array([fold_of(c, args.folds) for c in grp])
    for f in range(args.folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit((X[tr] - mu) / sd, y[tr])
        oof[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]

    # Recall: does the top-scored peak per candidate match the correct one?
    def recall(scorer):
        top1 = top3 = 0
        i = 0
        by = {}
        for c in grp:
            by.setdefault(c, []).append(i)
            i += 1
        for c, idx in by.items():
            idx = np.array(idx)
            corr = np.array([y[k] for k in idx]).argmax()
            order = np.argsort(-scorer[idx], kind="stable")
            if order[0] == corr:
                top1 += 1
            if corr in order[:3]:
                top3 += 1
        n = len(by)
        return top1 / n, top3 / n

    r_model = recall(oof)
    r_ev = recall(X[:, 0])   # evidence_count
    r_area = recall(X[:, 1])  # area
    res = {
        "label_kind": label_kind,
        "n_labeled_candidates": n_labeled,
        "model_top1": r_model[0], "model_top3": r_model[1],
        "evidence_rank_top1": r_ev[0], "evidence_rank_top3": r_ev[1],
        "area_rank_top1": r_area[0], "area_rank_top3": r_area[1],
    }
    print(f"=== peak-selection model ({label_kind}) ===")
    print(f"  labelable candidates: {n_labeled}")
    print(f"  learned model : top1={r_model[0]:.3f} top3={r_model[1]:.3f}")
    print(f"  evidence-rank : top1={r_ev[0]:.3f} top3={r_ev[1]:.3f}")
    print(f"  area-rank     : top1={r_area[0]:.3f} top3={r_area[1]:.3f}")
    if not diann:
        print("  NOTE: weak self-label (current apex); measures agreement with the "
              "existing heuristic, not correctness. Supply --diann for the honest label.")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
