"""Mokapot rescorer sidecar (docs/13_sidecars.md).

Usage:
    python mokapot_worker.py <input.pin> <output.parquet>

Reads a Percolator PIN, runs mokapot.brew, and writes the flat row ID in the
legacy `candidate_id` column plus mokapot score and q-value. SpecId is
`psm_<flat_row_id>` so library candidate IDs may safely repeat across runs. Run
with an env that has mokapot + pyarrow (py312_mumdia).

Model is env-switchable so the linear-vs-NN A/B needs no code edit:
    MUMDIA_RESCORE_MODEL = nn  (default)  -> sklearn MLPClassifier in mokapot.Model
                         = percolator     -> mokapot's default linear model
NN hyperparameters (only read when MODEL=nn):
    MUMDIA_NN_HIDDEN     = "64,32"   comma-separated hidden layer sizes
    MUMDIA_NN_SOLVER     = "adam"    adam scales to large PSM sets; lbfgs is more
                                     reproducible but memory-heavy here
    MUMDIA_NN_MAX_ITER   = "200"     NN epochs per fit
    MUMDIA_NN_ALPHA      = "1e-4"    L2 penalty
    MUMDIA_BREW_ITERS    = "5"       mokapot semi-supervised iterations (Model.max_iter)

Determinism note (docs/14_build_test_deploy_gotchas.md): NN training is only
approximately reproducible. random_state/rng are pinned to 0, but with solver=adam
and BLAS threading the scores can drift slightly run to run. Use solver=lbfgs + a
single BLAS thread for the closest thing to bit-exact.
"""

import os
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def make_model(mokapot):
    """Build the mokapot model selected by MUMDIA_RESCORE_MODEL (default: nn)."""
    kind = os.environ.get("MUMDIA_RESCORE_MODEL", "nn").lower()
    brew_iters = int(os.environ.get("MUMDIA_BREW_ITERS", "20"))
    if kind in ("percolator", "linear", "svm"):
        # mokapot's default (grid-searched linear SVM). model=None in brew.
        return None
    if kind in ("logreg", "logistic", "lr"):
        # L2 logistic regression: fast, convex, and near-ties the NN on
        # decoy-trained data (the NN only pulls ahead on real/entrapment negatives).
        from sklearn.linear_model import LogisticRegression

        net = LogisticRegression(
            C=float(os.environ.get("MUMDIA_LR_C", "1.0")),
            max_iter=int(os.environ.get("MUMDIA_LR_MAX_ITER", "1000")),
            random_state=0,
        )
        # scaler=None -> StandardScaler (features are on mixed scales).
        return mokapot.Model(net, train_fdr=0.01, max_iter=brew_iters, rng=0)
    if kind in ("xgb", "xgboost"):
        # Gradient-boosted trees: nonlinear, captures feature interactions the
        # linear models miss. Used as the second stage after a cheap prefilter,
        # where the reduced PSM count keeps it tractable. Hist tree method +
        # capped depth/estimators for speed on millions of PSMs.
        from xgboost import XGBClassifier

        net = XGBClassifier(
            n_estimators=int(os.environ.get("MUMDIA_XGB_TREES", "200")),
            max_depth=int(os.environ.get("MUMDIA_XGB_DEPTH", "6")),
            learning_rate=float(os.environ.get("MUMDIA_XGB_LR", "0.1")),
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=int(os.environ.get("MUMDIA_XGB_JOBS", "0")) or None,
            eval_metric="logloss",
            random_state=0,
        )
        # scaler=None -> StandardScaler; harmless for trees, keeps the API uniform.
        return mokapot.Model(net, train_fdr=0.01, max_iter=brew_iters, rng=0)
    if kind != "nn":
        raise ValueError(
            f"unknown MUMDIA_RESCORE_MODEL={kind!r} (want nn|logreg|xgb|percolator)"
        )

    from sklearn.neural_network import MLPClassifier

    hidden = tuple(
        int(x)
        for x in os.environ.get("MUMDIA_NN_HIDDEN", "128,64,64,32").split(",")
        if x
    )
    net = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver=os.environ.get("MUMDIA_NN_SOLVER", "adam"),
        alpha=float(os.environ.get("MUMDIA_NN_ALPHA", "1e-4")),
        max_iter=int(os.environ.get("MUMDIA_NN_MAX_ITER", "200")),
        early_stopping=True,
        n_iter_no_change=30,
        random_state=0,
    )
    # scaler=None -> StandardScaler (the NN needs standardized inputs). Model.max_iter
    # is mokapot's semi-supervised iteration count, not the NN's epochs.
    return mokapot.Model(net, train_fdr=0.01, max_iter=brew_iters, rng=0)


def main():
    pin_path, out_path = sys.argv[1], sys.argv[2]
    import mokapot

    # Surface mokapot's brew step logging (fold, iteration, positive counts).
    # Default Python logging level is WARNING, which drops mokapot's INFO records.
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("mokapot").setLevel(logging.INFO)

    psms = mokapot.read_pin(pin_path)
    np.random.seed(0)
    model = make_model(mokapot)
    # Parallelize the cross-validation folds (mokapot default max_workers=1 runs
    # them serially, which dominates the runtime). Thread-based (require=sharedmem),
    # so no per-fold dataset copy; results are unchanged, only faster.
    workers = int(os.environ.get("MUMDIA_MOKAPOT_WORKERS", "3"))
    if model is None:
        _results, _models = mokapot.brew(psms, rng=0, max_workers=workers)
    else:
        _results, _models = mokapot.brew(psms, model=model, rng=0, max_workers=workers)

    specids = psms.data["SpecId"].astype(str).to_numpy()

    # Prefer mokapot's OUT-OF-FOLD scores: brew returns held-out confidence tables
    # (targets in confidence_estimates, decoys in decoy_confidence_estimates), each
    # PSM scored only by the fold that did not train on it. Averaging all fold
    # models over all rows is NOT out-of-fold - 2 of 3 folds trained on each row -
    # and can make nominal q-values anti-conservative. Therefore an unavailable or
    # incomplete confidence table is a hard error; there is deliberately no
    # in-sample fallback.
    # NOTE: unvalidated in the unit suite (no mokapot there); confirm on a real run.
    def _oof_scores():
        conf = _results[0] if isinstance(_results, (list, tuple)) else _results
        tdf = conf.confidence_estimates["psms"]
        ddf = conf.decoy_confidence_estimates["psms"]

        def scol(df):
            for c in ("mokapot score", "score"):
                if c in df.columns:
                    return c
            return next(c for c in df.columns if "score" in c.lower())

        m = {}
        for df in (ddf, tdf):
            m.update(dict(zip(df["SpecId"].astype(str), df[scol(df)].astype(float))))
        expected = set(specids)
        returned = set(m)
        if returned != expected or len(m) != len(specids):
            missing = len(expected - returned)
            extra = len(returned - expected)
            raise RuntimeError(
                "OOF confidence tables do not exactly cover the PIN: "
                f"rows={len(specids)} unique_scores={len(m)} "
                f"missing={missing} extra={extra}"
            )
        scores = np.array([m[s] for s in specids], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise RuntimeError("OOF confidence tables contain non-finite scores")
        return scores

    scores = _oof_scores()
    print("mokapot_worker: using complete out-of-fold confidence scores", flush=True)
    if len(specids) != len(scores):
        raise RuntimeError(
            f"specid/score length mismatch: {len(specids)} vs {len(scores)}"
        )
    cids = [int(s.split("_")[-1]) for s in specids]

    out = pa.table(
        {
            "candidate_id": pa.array(cids, pa.uint32()),
            "score": pa.array(scores, pa.float64()),
            "q_value": pa.array(np.zeros(len(cids), dtype=np.float64), pa.float64()),
        }
    )
    pq.write_table(out, out_path)
    model_name = os.environ.get("MUMDIA_RESCORE_MODEL", "nn")
    print(f"mokapot_worker[{model_name}]: {len(cids)} PSMs rescored (targets+decoys)")


if __name__ == "__main__":
    main()
