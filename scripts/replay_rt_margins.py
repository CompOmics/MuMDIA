#!/usr/bin/env python3
"""Replay adaptive RT-margin calculation from a saved pre-margin dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mumdia import (
    add_retention_time_margins,
    calculate_min_max_margins,
    ensure_fragment_name_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        required=True,
        help="Directory containing the RT-margin input dump files.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write replayed margin outputs. Defaults to <dump-dir>/replayed_margins.",
    )
    parser.add_argument(
        "--fragment-table",
        choices=["auto", "reannotated", "search"],
        default="auto",
        help="Which dumped fragment table to use. 'auto' follows the table recorded in dump metadata when available.",
    )
    parser.add_argument(
        "--fragment-max-peptide-path",
        help="Optional override for the apex table used during margin replay.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Top N peptidoforms used for min/max RT-margin calibration. Defaults to dump metadata or 100.",
    )
    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=None,
        help="Relative intensity threshold used when walking away from the apex. Defaults to dump metadata or 0.05.",
    )
    parser.add_argument(
        "--min-diff",
        type=float,
        default=None,
        help="Override calibrated minimum RT half-width.",
    )
    parser.add_argument(
        "--max-diff",
        type=float,
        default=None,
        help="Override calibrated maximum RT half-width.",
    )
    parser.add_argument(
        "--use-global-ms2-rt-grid",
        action="store_true",
        help="Use the dumped all-MS2 RT grid for zero-fill. Off by default because it can include unrelated scans.",
    )
    return parser.parse_args()


def read_metadata(dump_dir: Path) -> dict[str, Any]:
    metadata_path = dump_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def load_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path, separator="\t")


def load_ms2_rt_grid(dump_dir: Path) -> np.ndarray | None:
    rt_grid_path = dump_dir / "global_ms2_rt_values.tsv"
    if not rt_grid_path.exists():
        return None

    df_rt = load_table(rt_grid_path)
    if df_rt.is_empty() or "rt" not in df_rt.columns:
        return None

    return df_rt["rt"].to_numpy().astype(np.float64)


def choose_fragment_file(
    dump_dir: Path, fragment_table: str, metadata: dict[str, Any]
) -> Path:
    reannotated = dump_dir / "df_fragment_reannotated_pre_margin.tsv"
    search = dump_dir / "df_fragment_search_pre_margin.tsv"

    if fragment_table == "reannotated":
        return reannotated
    if fragment_table == "search":
        return search
    metadata_choice = metadata.get("fragment_table_used_for_margin_calculation")
    if metadata_choice == reannotated.name and reannotated.exists():
        return reannotated
    if metadata_choice == search.name and search.exists():
        return search
    if reannotated.exists():
        return reannotated
    return search


def main() -> None:
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_dir}")

    metadata = read_metadata(dump_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else dump_dir / "replayed_margins"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df_psms = load_table(dump_dir / "df_psms_pre_margin.tsv")
    fragment_file = choose_fragment_file(dump_dir, args.fragment_table, metadata)
    df_fragment = ensure_fragment_name_column(load_table(fragment_file))

    df_fragment_max_peptide_path = (
        Path(args.fragment_max_peptide_path)
        if args.fragment_max_peptide_path
        else dump_dir / "df_fragment_max_peptide_pre_margin.tsv"
    )
    df_fragment_max_peptide = (
        load_table(df_fragment_max_peptide_path)
        if df_fragment_max_peptide_path.exists()
        else None
    )

    top_n = (
        args.top_n
        if args.top_n is not None
        else int(metadata.get("default_top_n", 100))
    )
    intensity_threshold = (
        args.intensity_threshold
        if args.intensity_threshold is not None
        else float(metadata.get("default_intensity_threshold", 0.05))
    )
    preferred_fragment_count = int(metadata.get("default_preferred_fragment_count", 2))
    global_ms2_rt_values = (
        load_ms2_rt_grid(dump_dir) if args.use_global_ms2_rt_grid else None
    )

    if args.min_diff is None or args.max_diff is None:
        calibrated_min_diff, calibrated_max_diff = calculate_min_max_margins(
            df_psms,
            df_fragment,
            df_fragment_max_peptide,
            top_n=top_n,
            intensity_threshold=intensity_threshold,
            preferred_fragment_count=preferred_fragment_count,
            global_ms2_rt_values=global_ms2_rt_values,
        )
    else:
        calibrated_min_diff = float(args.min_diff)
        calibrated_max_diff = float(args.max_diff)

    min_diff = (
        float(args.min_diff)
        if args.min_diff is not None
        else float(calibrated_min_diff)
    )
    max_diff = (
        float(args.max_diff)
        if args.max_diff is not None
        else float(calibrated_max_diff)
    )

    df_psms_with_margins, df_fragment_with_margins = add_retention_time_margins(
        df_psms,
        df_fragment,
        min_diff=min_diff,
        max_diff=max_diff,
        intensity_threshold=intensity_threshold,
        df_fragment_max_peptide=df_fragment_max_peptide,
        preferred_fragment_count=preferred_fragment_count,
        global_ms2_rt_values=global_ms2_rt_values,
    )

    df_psms_with_margins.write_csv(
        output_dir / "df_psms_with_replayed_margins.tsv", separator="\t"
    )
    df_fragment_with_margins.write_csv(
        output_dir / "df_fragment_with_replayed_margins.tsv", separator="\t"
    )

    if df_fragment_max_peptide is not None and not df_fragment_max_peptide.is_empty():
        apex_margins = df_psms_with_margins.select(
            ["peptide", "charge", "rt_lower_margin", "rt_higher_margin"]
        ).unique(subset=["peptide", "charge"], maintain_order=True)
        df_fragment_max_peptide_with_margins = df_fragment_max_peptide.drop(
            [
                col
                for col in ["rt_lower_margin", "rt_higher_margin"]
                if col in df_fragment_max_peptide.columns
            ]
        ).join(apex_margins, on=["peptide", "charge"], how="left")
        df_fragment_max_peptide_with_margins.write_csv(
            output_dir / "df_fragment_max_peptide_with_replayed_margins.tsv",
            separator="\t",
        )

    summary = {
        "dump_dir": str(dump_dir),
        "fragment_file_used": fragment_file.name,
        "top_n": top_n,
        "intensity_threshold": intensity_threshold,
        "preferred_fragment_count": preferred_fragment_count,
        "used_global_ms2_rt_values": bool(
            args.use_global_ms2_rt_grid
            and global_ms2_rt_values is not None
            and global_ms2_rt_values.size > 0
        ),
        "global_ms2_rt_values_present": bool(
            global_ms2_rt_values is not None and global_ms2_rt_values.size > 0
        ),
        "calibrated_min_diff": float(calibrated_min_diff),
        "calibrated_max_diff": float(calibrated_max_diff),
        "min_diff_used": min_diff,
        "max_diff_used": max_diff,
        "df_psms_shape": list(df_psms_with_margins.shape),
        "df_fragment_shape": list(df_fragment_with_margins.shape),
    }
    (output_dir / "margin_replay_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"Loaded dump from {dump_dir}")
    print(f"Using fragment table: {fragment_file.name}")
    print(f"Calibrated min/max: {calibrated_min_diff}, {calibrated_max_diff}")
    print(f"Applied min/max: {min_diff}, {max_diff}")
    print(f"Wrote replay outputs to {output_dir}")


if __name__ == "__main__":
    main()
