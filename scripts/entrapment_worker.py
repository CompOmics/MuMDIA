"""Entrapment (spike-in negative) rescorer sidecar (docs/11_compete_rescore_fdr.md).

Usage:
    python entrapment_worker.py <input.parquet> <output.parquet> [folds]

Trains a gradient-boosted classifier that separates real target PSMs (positive)
from spike-in foreign-proteome PSMs (negative). The spike-in population, unlike
in-silico decoys, experiences the same chimeric DIA interference as real targets,
so it is an empirically valid negative set. Scores are produced OUT OF FOLD with
GroupKFold grouped by base peptide, so no peptide leaks between train and test
and the entrapment negatives are scored by a model that never saw them. Decoy
rows are scored by a final model fit on all non-decoy PSMs.

A training fold that holds a single class is an error, not a gap to fill: the
previous behaviour skipped such a fold and then scored its held-out rows with the
final model, which had been trained on those very rows, so the confidence estimate
downstream rested on in-sample scores (docs/29 #3).

Input columns: candidate_id, base_peptide_id, is_entrapment (0/1), is_decoy
(0/1), and one column per feature. Output columns: candidate_id, score. Run with
an env that has scikit-learn + pyarrow (py312_mumdia).
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


def _new_model():
    # Classifier is env-switchable so the GBM-vs-NN comparison needs no code edit:
    #   MUMDIA_ENTRAPMENT_MODEL = gbm (default) | nn
    # Both train on the SAME real-target-vs-entrapment(human) negatives, so the NN
    # learns from empirically valid negatives (not in-silico decoys) - the regime
    # where a flexible model actually helps (AUC ~0.97 vs ~0.62 on decoys).
    kind = os.environ.get("MUMDIA_ENTRAPMENT_MODEL", "gbm").lower()
    if kind == "nn":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # StandardScaler is fit inside each fold's pipeline (no leakage). The MLP
        # needs standardized inputs; the 330-feature extended battery is on mixed
        # scales. predict_proba works through the pipeline.
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=0,
            ),
        )
    # Histogram-based GBM (default): fast and scales to hundreds of features (the
    # extended battery), unlike the exact GradientBoostingClassifier.
    # early_stopping=False removes the random validation split so a fixed
    # random_state is reproducible.
    return HistGradientBoostingClassifier(random_state=0, early_stopping=False)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    folds = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    t = pq.read_table(inp).to_pandas()
    meta = ["row_id", "candidate_id", "base_peptide_id", "is_entrapment", "is_decoy"]
    feat_cols = [c for c in t.columns if c not in meta]

    X = np.nan_to_num(t[feat_cols].to_numpy(dtype=np.float64), posinf=0.0, neginf=0.0)
    cid = t["candidate_id"].to_numpy()
    # Unique flat row id for collision-free score readback (candidate_id repeats
    # across competed runs). Fall back to positional index if absent.
    rid = t["row_id"].to_numpy() if "row_id" in t.columns else np.arange(len(t), dtype=np.uint32)
    grp = t["base_peptide_id"].to_numpy()
    is_ent = t["is_entrapment"].to_numpy() != 0
    is_dec = t["is_decoy"].to_numpy() != 0

    # y = 1 for real targets, 0 for entrapment; decoys excluded from training.
    train = ~is_dec
    y = np.where(is_ent, 0, 1).astype(int)
    Xt, yt, gt = X[train], y[train], grp[train]
    idx = np.where(train)[0]

    if len(np.unique(yt)) < 2:
        raise SystemExit("entrapment_worker: need both real-target and entrapment PSMs to train")

    n_groups = len(np.unique(gt))
    k = max(2, min(folds, n_groups))

    scores = np.full(len(t), np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=k)
    for fold_no, (tr, te) in enumerate(gkf.split(Xt, yt, gt)):
        classes = np.unique(yt[tr])
        if len(classes) < 2:
            raise SystemExit(
                f"entrapment_worker: training fold {fold_no + 1} of {k} holds a single "
                f"class ({'real targets' if classes[0] == 1 else 'entrapment'} only). Every "
                "training fold needs both real-target and entrapment PSMs: use more base "
                "peptides, fewer folds, or a larger spike-in library. Scoring the held-out "
                "rows with a model trained on them would inflate the entrapment FDR, so the "
                "worker refuses instead of filling the gap."
            )
        m = _new_model()
        m.fit(Xt[tr], yt[tr])
        scores[idx[te]] = m.predict_proba(Xt[te])[:, 1]

    # Every non-decoy row now carries an out-of-fold score; anything else is a fold
    # construction bug, not a data condition.
    gap = np.isnan(scores) & train
    if gap.any():
        raise SystemExit(
            f"entrapment_worker: {int(gap.sum())} non-decoy rows received no out-of-fold "
            "score; GroupKFold did not cover the training set"
        )

    # Final model on all non-decoy PSMs scores the decoys, which took no part in
    # training and have no fold.
    mf = _new_model()
    mf.fit(Xt, yt)
    dec_idx = np.where(is_dec)[0]
    if len(dec_idx):
        scores[dec_idx] = mf.predict_proba(X[dec_idx])[:, 1]

    out = pa.table({
        "row_id": pa.array(rid.astype("uint32")),
        "candidate_id": pa.array(cid.astype("uint32")),
        "score": pa.array(scores.astype(np.float64)),
    })
    pq.write_table(out, outp)
    print(
        f"entrapment_worker: {len(cid)} PSMs scored, "
        f"{int(is_ent.sum())} entrapment negatives, {int((train & (y == 1)).sum())} real positives, k={k}"
    )


if __name__ == "__main__":
    main()
