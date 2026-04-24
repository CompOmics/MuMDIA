#!/usr/bin/env python3
"""Inspect and plot DIA-NN XIC parquet files.

This script is intentionally schema-tolerant because DIA-NN XIC parquet layouts
can vary between versions/workflows. It can:

- print the parquet schema and preview rows
- detect likely RT / intensity / fragment / precursor / run columns
- handle both row-wise and list-column XIC layouts
- plot one precursor/run to a PNG

Examples
--------
Inspect a file:
    python scripts/plot_diann_xic.py --xic path/to/run.xic.parquet --inspect

Plot a precursor:
    python scripts/plot_diann_xic.py \
        --xic path/to/run.xic.parquet \
        --precursor "PEPTIDEK2" \
        --out xic.png

Plot using a stripped sequence match:
    python scripts/plot_diann_xic.py \
        --xic path/to/run.xic.parquet \
        --stripped-sequence PEPTIDEK \
        --run sample1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import polars as pl

COLUMN_CANDIDATES = {
    "run": ["Run", "File.Name", "File", "Raw.File", "RawName"],
    "precursor": [
        "pr",
        "Precursor.Id",
        "Modified.Sequence",
        "Precursor",
        "Peptide",
        "Peptidoform",
    ],
    "stripped_sequence": ["Stripped.Sequence", "Stripped", "Sequence"],
    "fragment": [
        "feature",
        "Fragment.Id",
        "Fragment",
        "FragmentName",
        "Transition",
        "Product",
    ],
    "rt": ["RT", "Rt", "rt", "Retention.Time", "RetentionTime", "Time"],
    "intensity": [
        "value",
        "Intensity",
        "Fragment.Intensity",
        "Signal",
        "Area",
        "Value",
        "Y",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xic", required=True, help="Path to DIA-NN .xic.parquet")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print schema, inferred columns, and preview rows",
    )
    parser.add_argument(
        "--preview-rows", type=int, default=5, help="Rows to preview during inspect"
    )
    parser.add_argument("--run", help="Filter by run name (substring match)")
    parser.add_argument(
        "--precursor", help="Filter by precursor / modified sequence (substring match)"
    )
    parser.add_argument(
        "--stripped-sequence", help="Filter by stripped sequence (substring match)"
    )
    parser.add_argument(
        "--fragment-substring",
        help="Optional fragment filter, e.g. y7 or b3",
    )
    parser.add_argument(
        "--top-fragments",
        type=int,
        default=12,
        help="Maximum number of fragment traces to draw",
    )
    parser.add_argument(
        "--out",
        default="diann_xic_plot.png",
        help="Output image path for the plot",
    )
    return parser.parse_args()


def find_column(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def infer_columns(df: pl.DataFrame) -> dict[str, str | None]:
    return {
        name: find_column(df.columns, candidates)
        for name, candidates in COLUMN_CANDIDATES.items()
    }


def dtype_is_list(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.List)


def _string_filter(
    df: pl.DataFrame, column: str | None, value: str | None
) -> pl.DataFrame:
    if column is None or value is None:
        return df
    return df.filter(pl.col(column).cast(pl.String).str.contains(value, literal=True))


def normalize_xic_layout(
    df: pl.DataFrame, inferred: dict[str, str | None]
) -> pl.DataFrame:
    """Return a long-form table with one RT/intensity point per row."""
    rt_col = inferred["rt"]
    intensity_col = inferred["intensity"]
    if rt_col is None or intensity_col is None:
        raise ValueError(
            "Could not infer RT/intensity columns. Run with --inspect to review schema."
        )

    rt_dtype = df.schema[rt_col]
    int_dtype = df.schema[intensity_col]

    if dtype_is_list(rt_dtype) and dtype_is_list(int_dtype):
        id_columns = [
            col
            for key, col in inferred.items()
            if key not in {"rt", "intensity"} and col is not None
        ]
        return df.explode([rt_col, intensity_col]).select(
            id_columns + [rt_col, intensity_col]
        )

    return df


def inspect(
    df: pl.DataFrame, inferred: dict[str, str | None], preview_rows: int
) -> None:
    print("Schema:")
    for name, dtype in df.schema.items():
        print(f"  {name}: {dtype}")

    print("\nInferred columns:")
    for key, value in inferred.items():
        print(f"  {key}: {value}")

    print(f"\nPreview ({preview_rows} rows):")
    print(df.head(preview_rows))

    if inferred["precursor"] is not None:
        print("\nExample precursor values:")
        print(df.select(pl.col(inferred["precursor"])).unique().head(10))
    if inferred["run"] is not None:
        print("\nExample run values:")
        print(df.select(pl.col(inferred["run"])).unique().head(10))


def choose_fragment_column(df: pl.DataFrame, inferred: dict[str, str | None]) -> str:
    fragment_col = inferred["fragment"]
    if fragment_col is not None:
        return fragment_col
    fallback = "__fragment__"
    return fallback if fallback in df.columns else ""


def main() -> None:
    args = parse_args()
    xic_path = Path(args.xic)
    if not xic_path.exists():
        raise FileNotFoundError(f"XIC parquet not found: {xic_path}")

    df = pl.read_parquet(xic_path)
    inferred = infer_columns(df)

    if args.inspect:
        inspect(df, inferred, args.preview_rows)

    # If only inspection was requested, stop here unless filters were also provided.
    if args.inspect and not any(
        [args.run, args.precursor, args.stripped_sequence, args.fragment_substring]
    ):
        return

    df = _string_filter(df, inferred["run"], args.run)
    df = _string_filter(df, inferred["precursor"], args.precursor)
    df = _string_filter(df, inferred["stripped_sequence"], args.stripped_sequence)

    if df.is_empty():
        raise ValueError("No rows left after applying run/precursor filters.")

    df = normalize_xic_layout(df, inferred)
    rt_col = inferred["rt"]
    intensity_col = inferred["intensity"]
    if rt_col is None or intensity_col is None:
        raise ValueError("Could not infer RT/intensity columns.")

    fragment_col = choose_fragment_column(df, inferred)
    if not fragment_col:
        fragment_col = "__fragment__"
        df = df.with_columns(pl.lit("trace").alias(fragment_col))

    if args.fragment_substring:
        df = _string_filter(df, fragment_col, args.fragment_substring)
        if df.is_empty():
            raise ValueError("No rows left after applying fragment filter.")

    # Keep the most populated fragments for readability.
    frag_counts = (
        df.group_by(fragment_col)
        .len()
        .sort("len", descending=True)
        .head(args.top_fragments)
    )
    keep_fragments = frag_counts[fragment_col].to_list()
    df = df.filter(pl.col(fragment_col).is_in(keep_fragments))

    pdf = df.select([fragment_col, rt_col, intensity_col]).to_pandas()
    pdf = pdf.sort_values([fragment_col, rt_col])

    plt.figure(figsize=(10, 6))
    for frag, frag_df in pdf.groupby(fragment_col):
        plt.plot(
            frag_df[rt_col],
            frag_df[intensity_col],
            marker="o",
            linewidth=1.5,
            label=str(frag),
        )

    title_parts = [xic_path.name]
    if args.run:
        title_parts.append(f"run={args.run}")
    if args.precursor:
        title_parts.append(f"precursor={args.precursor}")
    if args.stripped_sequence:
        title_parts.append(f"sequence={args.stripped_sequence}")

    plt.title(" | ".join(title_parts))
    plt.xlabel(rt_col)
    plt.ylabel(intensity_col)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
