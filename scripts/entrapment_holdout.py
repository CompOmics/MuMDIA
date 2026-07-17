"""Held-out entrapment validation harness.

The entrapment-trained scorer uses the human proteome as negatives, but training,
FDR, and reporting on the SAME human population overfits (research leak). This
harness splits the human entrapment proteins into two disjoint halves by a stable
protein hash, trains the classifier on E. coli targets (positive) vs human-TRAIN
(negative), and then evaluates FDR/sensitivity against the UNSEEN human-TEST null.
That held-out E. coli count at 1% is the honest number.

Usage:
  python entrapment_holdout.py <competed_or_scored.parquet> [--folds 3] [--q 0.01]

Requires an env with scikit-learn + pyarrow (py312_mumdia). Deterministic.
"""
import hashlib
import re
import sys

import numpy as np
import pyarrow.dataset as pds
from sklearn.ensemble import HistGradientBoostingClassifier

strip = lambda p: re.sub(r"\[[^\]]*\]", "", str(p))


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def half(protein: str) -> int:
    # Stable (non-randomized) split of a protein string into half 0/1.
    h = hashlib.md5(str(protein).encode("utf-8")).hexdigest()
    return int(h, 16) & 1


def ent_q(score, is_entrap, is_real, ratio):
    order = np.argsort(-score, kind="stable")
    ne = nr = 0
    fdr = np.ones(len(score))
    for rank, i in enumerate(order):
        if is_entrap[i]:
            ne += 1
        elif is_real[i]:
            nr += 1
        fdr[rank] = ratio * ne / max(1, nr)
    q = np.ones(len(score))
    qmin = 1.0
    for rank in range(len(score) - 1, -1, -1):
        qmin = min(qmin, fdr[rank])
        q[order[rank]] = qmin
    return q


def main():
    path = sys.argv[1]
    q_cut = float(arg("--q", "0.01"))
    t = pds.dataset(path).to_table().to_pandas()
    pcol = "protein_group" if "protein_group" in t.columns else "protein"
    dec = t.label.eq("decoy").to_numpy()
    human = (t[pcol].str.contains("_HUMAN") & ~t[pcol].str.contains("_ECOLI")).to_numpy()
    ecoli_tgt = (~dec) & (~human)
    hh = t[pcol].map(half).to_numpy()
    train_neg = human & (hh == 0)  # human-TRAIN negatives
    test_neg = human & (hh == 1)   # human-TEST null (unseen)

    meta = {"candidate_id", "label", "peptidoform", "protein", "protein_group",
            "base_peptide_id", "charge", "q_value", "peptide_q_value", "pg_q_value",
            "global_q_value", "score", "prelim_score", "source"}
    feats = [c for c in t.columns if c not in meta and np.issubdtype(t[c].dtype, np.number)]
    X = np.nan_to_num(t[feats].to_numpy(np.float64), posinf=0.0, neginf=0.0)

    # Train ONLY on E. coli targets (pos) vs human-TRAIN (neg); human-TEST + decoys
    # are never seen in training.
    tr = ecoli_tgt | train_neg
    y = np.where(ecoli_tgt, 1, 0)[tr]
    if len(np.unique(y)) < 2:
        raise SystemExit("entrapment_holdout: need both E.coli targets and human-train negatives")
    m = HistGradientBoostingClassifier(random_state=0, early_stopping=False)
    m.fit(X[tr], y)
    score = m.predict_proba(X)[:, 1]

    # Held-out evaluation: null = human-TEST (unseen). Library-size ratio correction.
    ratio = int(ecoli_tgt.sum()) / max(1, int(test_neg.sum()))
    q = ent_q(score, test_neg, ecoli_tgt, ratio)
    gate = q <= q_cut
    eco_heldout = t.loc[ecoli_tgt & gate, "peptidoform"].map(strip).nunique()

    # For contrast: FDR at the shipped q_value cutoff, measured on the held-out null.
    shipped = None
    if "q_value" in t.columns:
        g2 = t.q_value.to_numpy() <= q_cut
        eco_s = t.loc[ecoli_tgt & g2, "peptidoform"].map(strip).nunique()
        leak = int((test_neg & g2).sum())
        shipped = (eco_s, ratio * leak / max(1, int((ecoli_tgt & g2).sum())) * 100)

    print(f"=== held-out entrapment ({path}) ===")
    print(f"  E.coli targets={int(ecoli_tgt.sum())}  human train-neg={int(train_neg.sum())}  "
          f"test-null={int(test_neg.sum())}  ratio={ratio:.3f}")
    print(f"  held-out E.coli stripped seqs @ {q_cut:.0%} (unseen null): {eco_heldout}")
    if shipped:
        print(f"  at shipped q<= {q_cut:.0%}: E.coli={shipped[0]}, true FDR on held-out null = {shipped[1]:.2f}%")


if __name__ == "__main__":
    main()
