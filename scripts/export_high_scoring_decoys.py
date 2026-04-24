#!/usr/bin/env python3
"""Export high-scoring decoys from a MuMDIA PIN file using mokapot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mumdia import create_model


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
            "comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s/"
            "high_scoring_decoys"
        ),
        help="Directory where mokapot decoy exports and candidate TSVs are written.",
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=0.10,
        help="Keep decoys with mokapot q-value at or below this threshold.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run mokapot even if exported decoy scores already exist.",
    )
    return parser.parse_args()


def load_or_run_mokapot(pin_file: Path, output_dir: Path, force_rerun: bool) -> Path:
    mokapot_psm_path = output_dir / "decoys.mokapot.decoy.psms.txt"
    if mokapot_psm_path.exists() and not force_rerun:
        return mokapot_psm_path

    try:
        import mokapot
        from scikeras.wrappers import KerasClassifier
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Failed to import mokapot/scikeras: {exc}") from exc

    psms = mokapot.read_pin(str(pin_file))
    model = KerasClassifier(
        build_fn=create_model,
        epochs=100,
        batch_size=1000,
        verbose=10,
    )
    results, _models = mokapot.brew(psms, mokapot.Model(model), folds=3)
    results.to_txt(dest_dir=str(output_dir), file_root="decoys", decoys=True)
    if not mokapot_psm_path.exists():
        raise FileNotFoundError(
            f"Expected mokapot output not found: {mokapot_psm_path}"
        )
    return mokapot_psm_path


def to_bool(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["true", "1", "target", "+1"])


def main() -> None:
    args = parse_args()
    pin_file = Path(args.pin_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mokapot_psm_path = load_or_run_mokapot(pin_file, output_dir, args.force_rerun)
    mokapot_df = pd.read_csv(mokapot_psm_path, sep="\t", low_memory=False)
    pin_df = pd.read_csv(
        pin_file,
        sep="\t",
        low_memory=False,
        usecols=["ScanNr", "filename", "Peptide", "charge", "SpecId", "Label"],
    )

    label_col = "Label"
    qval_col = "mokapot q-value"
    if label_col not in mokapot_df.columns or qval_col not in mokapot_df.columns:
        raise KeyError(
            f"Expected columns '{label_col}' and '{qval_col}' in {mokapot_psm_path}"
        )

    mokapot_df[qval_col] = pd.to_numeric(mokapot_df[qval_col], errors="coerce")
    mokapot_df["is_target"] = to_bool(mokapot_df[label_col])
    decoy_df = mokapot_df[
        (~mokapot_df["is_target"]) & (mokapot_df[qval_col] <= float(args.q_threshold))
    ].copy()

    decoy_candidates = decoy_df.merge(
        pin_df[["ScanNr", "filename", "Peptide", "charge", "SpecId"]],
        on=["ScanNr", "filename", "Peptide"],
        how="left",
    )
    decoy_candidates = decoy_candidates.dropna(subset=["charge"]).copy()
    decoy_candidates["charge"] = pd.to_numeric(
        decoy_candidates["charge"], errors="coerce"
    ).round()
    decoy_candidates = decoy_candidates.dropna(subset=["charge"]).copy()
    decoy_candidates["charge"] = decoy_candidates["charge"].astype(int)
    decoy_candidates = decoy_candidates.sort_values(
        [qval_col, "mokapot score"], ascending=[True, False], na_position="last"
    )

    candidate_path = output_dir / f"high_scoring_decoys_q{args.q_threshold:.2f}.tsv"
    decoy_candidates.to_csv(candidate_path, sep="\t", index=False)

    print(f"Mokapot export: {mokapot_psm_path}")
    print(f"High-scoring decoys: {len(decoy_candidates)}")
    print(f"Candidate TSV: {candidate_path}")


if __name__ == "__main__":
    main()
