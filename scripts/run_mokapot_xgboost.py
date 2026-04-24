#!/usr/bin/env python3
"""Run mokapot with an XGBoost classifier on a PIN file."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pin-file",
        default="results_ecoli_rerun_20260402_py312_rust_deeplc4_60s/outfile.pin",
        help="PIN file used for mokapot scoring.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results_ecoli_rerun_20260402_py312_rust_deeplc4_60s/" "mokapot_xgboost"
        ),
        help="Directory where mokapot outputs are written.",
    )
    parser.add_argument(
        "--file-root",
        default="xgboost",
        help="Prefix for mokapot output files.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of cross-validation folds for mokapot.",
    )
    parser.add_argument(
        "--train-fdr",
        type=float,
        default=0.01,
        help="Training-set FDR used by mokapot.",
    )
    parser.add_argument(
        "--test-fdr",
        type=float,
        default=0.01,
        help="Report the number of accepted target PSMs at this q-value.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of boosting rounds.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Boosting learning rate.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsampling fraction.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Feature subsampling fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=1,
        help="Random seed for XGBoost.",
    )
    parser.add_argument(
        "--write-decoys",
        action="store_true",
        help="Also export mokapot decoy outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd
    import mokapot
    from xgboost import XGBClassifier

    pin_file = Path(args.pin_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    psms = mokapot.read_pin(str(pin_file))
    model = XGBClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(args.random_state),
        n_jobs=1,
    )
    results, models = mokapot.brew(
        psms,
        mokapot.Model(model, train_fdr=float(args.train_fdr)),
        folds=int(args.folds),
    )
    result_files = results.to_txt(
        dest_dir=str(output_dir),
        file_root=str(args.file_root),
        decoys=bool(args.write_decoys),
    )

    target_psm_path = output_dir / f"{args.file_root}.mokapot.psms.txt"
    if not target_psm_path.exists():
        raise FileNotFoundError(f"Expected mokapot output not found: {target_psm_path}")

    psm_df = pd.read_csv(target_psm_path, sep="\t", low_memory=False)
    qvals = pd.to_numeric(psm_df.get("mokapot q-value"), errors="coerce")
    accepted = int((qvals <= float(args.test_fdr)).sum())

    print(f"PIN file: {pin_file}")
    print(f"Output directory: {output_dir}")
    print("Model: XGBClassifier")
    print(f"Folds: {args.folds}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth: {args.max_depth}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"Accepted target PSMs at q<={args.test_fdr}: {accepted}")
    print("Result files:")
    if isinstance(result_files, (list, tuple)):
        for path in result_files:
            print(path)
    else:
        print(result_files)
    print(f"Trained models: {len(models)}")


if __name__ == "__main__":
    main()
