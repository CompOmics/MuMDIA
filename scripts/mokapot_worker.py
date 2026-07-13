"""Mokapot rescorer sidecar (PLAN.md Section 0, Stage F).

Usage:
    python mokapot_worker.py <input.pin> <output.parquet>

Reads a Percolator PIN, runs mokapot.brew, writes candidate_id + mokapot score
+ q-value. SpecId is `cand_<candidate_id>`. Run with an env that has mokapot +
pyarrow (py312_mumdia).

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

Determinism note (plan.md Section 7): NN training is only approximately
reproducible. random_state/rng are pinned to 0, but with solver=adam and BLAS
threading the scores can drift slightly run to run. Use solver=lbfgs + a single
BLAS thread for the closest thing to bit-exact.
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
    if kind != "nn":
        raise ValueError(
            f"unknown MUMDIA_RESCORE_MODEL={kind!r} (want nn|logreg|percolator)"
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
    if model is None:
        results, _models = mokapot.brew(psms, rng=0)
    else:
        results, _models = mokapot.brew(psms, model=model, rng=0)

    # mokapot >=0.9 exposes PSM-level confidence as a DataFrame.
    df = None
    for attr in ("psms", "confidence_estimates"):
        obj = getattr(results, attr, None)
        if obj is not None:
            df = obj["psms"] if isinstance(obj, dict) else obj
            break
    if df is None and hasattr(results, "to_df"):
        df = results.to_df()
    if df is None:
        raise RuntimeError("could not extract mokapot PSM results")

    cols = {c.lower(): c for c in df.columns}
    spec_col = cols.get("specid") or cols.get("psmid") or list(df.columns)[0]
    score_col = next(c for c in df.columns if "score" in c.lower())
    q_col = next(
        c for c in df.columns if "q-value" in c.lower() or "q_value" in c.lower()
    )

    specids = df[spec_col].astype(str).tolist()
    cids = [int(s.split("_")[-1]) for s in specids]
    scores = df[score_col].to_numpy(dtype=np.float64)
    qs = df[q_col].to_numpy(dtype=np.float64)

    out = pa.table(
        {
            "candidate_id": pa.array(cids, pa.uint32()),
            "score": pa.array(scores, pa.float64()),
            "q_value": pa.array(qs, pa.float64()),
        }
    )
    pq.write_table(out, out_path)
    model_name = os.environ.get("MUMDIA_RESCORE_MODEL", "nn")
    print(f"mokapot_worker[{model_name}]: {len(cids)} PSMs rescored")


if __name__ == "__main__":
    main()
