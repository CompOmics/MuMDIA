"""Extract DIA-NN XIC traces from parquet to a reusable long-form table.

This utility is intentionally schema-tolerant. It can:
- infer likely DIA-NN XIC columns (run/precursor/fragment/RT/intensity),
- filter by run, precursor, sequence, and fragment,
- normalize list-column layouts to one point per row,
- export the traces to TSV,
- optionally render a quick line plot.

Examples
--------
Inspect schema/columns:
    python scripts/diann_parquet_to_xic.py --xic path/to/file.xic.parquet --inspect

Export one precursor XIC table:
    python scripts/diann_parquet_to_xic.py \
        --xic path/to/file.xic.parquet \
        --precursor "PEPTIDE(UniMod:4)2" \
        --out-tsv xic_points.tsv

Export + plot:
    python scripts/diann_parquet_to_xic.py \
        --xic path/to/file.xic.parquet \
        --stripped-sequence PEPTIDE \
        --out-tsv xic_points.tsv \
        --out-plot xic_plot.png
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
        help="Maximum number of fragment traces to keep",
    )
    parser.add_argument(
        "--out-tsv",
        default="diann_xic_points.tsv",
        help="Output TSV path for long-form XIC points",
    )
    parser.add_argument(
        "--out-plot",
        default="diann_xic_plot.png",
        help="Output image path for an optional quick plot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot rendering and only export TSV",
    )
    return parser.parse_args()


def find_column(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
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
    rt_col = inferred["rt"]
    intensity_col = inferred["intensity"]
    if rt_col is None or intensity_col is None:
        raise ValueError(
            "Could not infer RT/intensity columns. Run with --inspect to review schema."
        )

    rt_dtype = df.schema[rt_col]
    intensity_dtype = df.schema[intensity_col]

    if dtype_is_list(rt_dtype) and dtype_is_list(intensity_dtype):
        id_columns = [
            column
            for key, column in inferred.items()
            if key not in {"rt", "intensity"} and column is not None
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


def to_canonical_xic_table(
    df: pl.DataFrame, inferred: dict[str, str | None]
) -> pl.DataFrame:
    rt_col = inferred["rt"]
    intensity_col = inferred["intensity"]
    if rt_col is None or intensity_col is None:
        raise ValueError("Could not infer RT/intensity columns.")

    fragment_col = inferred["fragment"]
    if fragment_col is None:
        fragment_col = "__fragment__"
        if fragment_col not in df.columns:
            df = df.with_columns(pl.lit("trace").alias(fragment_col))

    run_col = inferred["run"]
    precursor_col = inferred["precursor"]
    stripped_col = inferred["stripped_sequence"]

    def maybe_column(column_name: str | None, alias_name: str) -> pl.Expr:
        if column_name is None or column_name not in df.columns:
            return pl.lit(None).cast(pl.String).alias(alias_name)
        return pl.col(column_name).cast(pl.String).alias(alias_name)

    return (
        df.select(
            [
                maybe_column(run_col, "run"),
                maybe_column(precursor_col, "precursor"),
                maybe_column(stripped_col, "stripped_sequence"),
                pl.col(fragment_col).cast(pl.String).alias("fragment"),
                pl.col(rt_col).cast(pl.Float64).alias("rt"),
                pl.col(intensity_col).cast(pl.Float64).alias("intensity"),
            ]
        )
        .drop_nulls(subset=["rt", "intensity"])
        .sort(["fragment", "rt"])
    )


def render_plot(df: pl.DataFrame, out_plot: str) -> None:
    pdf = df.to_pandas()
    plt.figure(figsize=(10, 6))
    for fragment, fragment_df in pdf.groupby("fragment"):
        plt.plot(
            fragment_df["rt"],
            fragment_df["intensity"],
            marker="o",
            linewidth=1.5,
            label=str(fragment),
        )
    plt.xlabel("rt")
    plt.ylabel("intensity")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    print(f"Saved plot to {out_plot}")


def main() -> None:
    args = parse_args()
    xic_path = Path(args.xic)
    if not xic_path.exists():
        raise FileNotFoundError(f"XIC parquet not found: {xic_path}")

    df = pl.read_parquet(xic_path)
    inferred = infer_columns(df)

    if args.inspect:
        inspect(df, inferred, args.preview_rows)

    df = _string_filter(df, inferred["run"], args.run)
    df = _string_filter(df, inferred["precursor"], args.precursor)
    df = _string_filter(df, inferred["stripped_sequence"], args.stripped_sequence)

    if df.is_empty():
        raise ValueError("No rows left after applying run/precursor/sequence filters.")

    df = normalize_xic_layout(df, inferred)

    fragment_col = inferred["fragment"]
    if fragment_col is not None and args.fragment_substring:
        df = _string_filter(df, fragment_col, args.fragment_substring)
        if df.is_empty():
            raise ValueError("No rows left after applying fragment filter.")

    canonical = to_canonical_xic_table(df, inferred)
    if canonical.is_empty():
        raise ValueError("No XIC points available after normalization/filtering.")

    fragment_counts = (
        canonical.group_by("fragment")
        .len()
        .sort("len", descending=True)
        .head(args.top_fragments)
    )
    keep_fragments = fragment_counts["fragment"].to_list()
    canonical = canonical.filter(pl.col("fragment").is_in(keep_fragments))

    canonical.write_csv(args.out_tsv, separator="\t")
    print(f"Saved XIC table to {args.out_tsv} ({canonical.height} points)")

    if not args.no_plot:
        render_plot(canonical, args.out_plot)


if __name__ == "__main__":
    main()
