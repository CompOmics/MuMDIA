#!/usr/bin/env python3
"""Run mokapot with a simple Percolator-style linear SVM model on a PIN file."""

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
            "results_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "mokapot_svm"
        ),
        help="Directory where mokapot outputs are written.",
    )
    parser.add_argument(
        "--file-root",
        default="svm",
        help="Prefix for mokapot output files.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of cross-validation folds for mokapot.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Maximum number of iterative semi-supervised updates.",
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
        "--write-decoys",
        action="store_true",
        help="Also export mokapot decoy outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd
    import mokapot

    pin_file = Path(args.pin_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    psms = mokapot.read_pin(str(pin_file))
    model = mokapot.PercolatorModel(
        train_fdr=float(args.train_fdr),
        max_iter=int(args.max_iter),
        override=True,
    )
    results, models = mokapot.brew(psms, model, folds=int(args.folds))
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
    print(f"Model: PercolatorModel (linear SVM)")
    print(f"Folds: {args.folds}")
    print(f"Iterations: {args.max_iter}")
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
