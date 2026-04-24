#!/usr/bin/env python3
"""Audit specific peptidoforms across MuMDIA stage-2 and downstream outputs.

This helper collects all rows for selected peptidoforms from the targeted-search
(stage 2) outputs and later debug/final outputs, writes small TSV extracts, and
creates quick XIC-style diagnostic plots.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

UNIMOD_REPLACEMENTS = {
    "[UniMod:4]": "[Carbamidomethyl]",
    "[UniMod:35]": "[Oxidation]",
}

DEFAULT_PEPTIDOFORMS = [
    "AAAEVGAPFIEIHTGC[UniMod:4]YADAK/3",
    "AVESVGGQLLITADHGNAEQM[UniMod:35]R/3",
    "AVIGVASC[UniMod:4]DK/2",
    "EVPVEVKPEVR/2",
    "GGVLAGEEEAESIVALAQR/3",
    "GSHIVVPR/2",
    "GVSLEVSQEAR/2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="results_ecoli_rerun_20260402_py312_rust_deeplc4_60s"
    )
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument(
        "--comparison-dir",
        default="comparison_output_ecoli_rerun_20260402_py312_rust_deeplc4_60s",
    )
    parser.add_argument("--output-dir", default="audit_peptidoforms")
    parser.add_argument(
        "--peptidoform",
        action="append",
        dest="peptidoforms",
        help="Peptidoform in peptide/charge form, e.g. PEPTIDE[UniMod:4]K/2",
    )
    parser.add_argument(
        "--peptidoform-file",
        help="TSV file containing suspicious/matched peptides. Uses Peptide+charge columns when available.",
    )
    return parser.parse_args()


def normalize_peptide(peptide: str) -> str:
    normalized = str(peptide).strip()
    for src, dst in UNIMOD_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)
    return normalized


def parse_peptidoform(text: str) -> tuple[str, int]:
    value = str(text).strip()
    if "/" not in value:
        raise ValueError(f"Expected peptidoform in peptide/charge form, got: {value}")
    peptide, charge = value.rsplit("/", 1)
    return normalize_peptide(peptide), int(charge)


def load_peptidoforms_from_file(path: str) -> list[str]:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    peptidoforms: list[str] = []

    if {"Peptide", "charge"}.issubset(df.columns):
        for row in (
            df[["Peptide", "charge"]].dropna().drop_duplicates().itertuples(index=False)
        ):
            peptidoforms.append(f"{row.Peptide}/{int(row.charge)}")
        return peptidoforms

    if "target_precursor" in df.columns:
        for value in df["target_precursor"].dropna().drop_duplicates().astype(str):
            match = re.match(r"^(.*?)(\d+)$", value.strip())
            if match:
                peptidoforms.append(f"{match.group(1)}/{int(match.group(2))}")
        return peptidoforms

    raise ValueError(
        "Peptidoform file must contain either Peptide+charge columns or target_precursor"
    )


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def ensure_fragment_name(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "fragment_name" not in out.columns and {
        "fragment_type",
        "fragment_ordinals",
        "fragment_charge",
    }.issubset(out.columns):
        out["fragment_name"] = (
            out["fragment_type"].astype(str)
            + out["fragment_ordinals"].astype(int).astype(str)
            + "^"
            + out["fragment_charge"].astype(int).astype(str)
        )
    return out


def canonical_fragment_name(name: str) -> str:
    return str(name).replace("^", "/")


def filter_peptidoform(df: pd.DataFrame, peptide: str, charge: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "peptide" not in out.columns or "charge" not in out.columns:
        return out.iloc[0:0].copy()
    return out[
        (out["peptide"] == peptide) & (out["charge"].astype(int) == charge)
    ].copy()


def filter_mokapot(df: pd.DataFrame, peptide: str) -> pd.DataFrame:
    if df.empty or "Peptide" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["Peptide"] == peptide].copy()


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, sep="\t", index=False)


def summarize_psms(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_rows": 0}
    summary: dict[str, object] = {
        "n_rows": int(len(df)),
        "n_psm_ids": int(df["psm_id"].nunique()) if "psm_id" in df.columns else None,
        "rt_min": float(df["rt"].min()) if "rt" in df.columns else None,
        "rt_max": float(df["rt"].max()) if "rt" in df.columns else None,
        "best_spectrum_q": (
            float(df["spectrum_q"].min()) if "spectrum_q" in df.columns else None
        ),
        "best_peptide_q": (
            float(df["peptide_q"].min()) if "peptide_q" in df.columns else None
        ),
        "best_fragment_intensity": (
            float(df["fragment_intensity"].max())
            if "fragment_intensity" in df.columns
            else None
        ),
    }
    for col in [
        "rt_predictions",
        "rt_prediction_error_abs",
        "rt_prediction_error_abs_relative",
        "rt_lower_margin",
        "rt_higher_margin",
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                summary[f"{col}_min"] = float(vals.min())
                summary[f"{col}_max"] = float(vals.max())
    return summary


def summarize_fragments(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_rows": 0}
    summary: dict[str, object] = {
        "n_rows": int(len(df)),
        "n_fragment_names": (
            int(df["fragment_name"].nunique())
            if "fragment_name" in df.columns
            else None
        ),
        "rt_min": float(df["rt"].min()) if "rt" in df.columns else None,
        "rt_max": float(df["rt"].max()) if "rt" in df.columns else None,
    }
    if "fragment_name" in df.columns:
        totals = (
            df.groupby("fragment_name", as_index=False)["fragment_intensity"]
            .sum()
            .sort_values("fragment_intensity", ascending=False)
        )
        summary["top_fragments"] = totals.head(10).to_dict("records")
    return summary


def get_rt_limits(
    df: pd.DataFrame, pad_fraction: float = 0.08
) -> tuple[float, float] | None:
    if df.empty or "rt" not in df.columns:
        return None
    rt_vals = pd.to_numeric(df["rt"], errors="coerce").dropna()
    if rt_vals.empty:
        return None
    rt_min = float(rt_vals.min())
    rt_max = float(rt_vals.max())
    span = max(rt_max - rt_min, 0.05)
    padding = max(span * pad_fraction, 0.02)
    return rt_min - padding, rt_max + padding


def get_margin_limits(
    df: pd.DataFrame, pad_fraction: float = 0.08
) -> tuple[float, float] | None:
    required = {"rt_lower_margin", "rt_higher_margin"}
    if df.empty or not required.issubset(df.columns):
        return None
    margins = (
        df[["rt_lower_margin", "rt_higher_margin"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    if margins.empty:
        return None
    left = float(margins["rt_lower_margin"].iloc[0])
    right = float(margins["rt_higher_margin"].iloc[0])
    span = max(right - left, 0.05)
    padding = max(span * pad_fraction, 0.02)
    return left - padding, right + padding


def plot_fragments(
    df: pd.DataFrame,
    path: Path,
    title: str,
    show_margins: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> None:
    if df.empty:
        return
    df = ensure_fragment_name(df)
    if (
        "fragment_name" not in df.columns
        or "rt" not in df.columns
        or "fragment_intensity" not in df.columns
    ):
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    fragment_names = list(df["fragment_name"].dropna().unique())
    colors = matplotlib.colormaps.get_cmap("tab20").resampled(
        max(len(fragment_names), 1)
    )
    color_map = {frag: colors(i) for i, frag in enumerate(fragment_names)}

    for frag in fragment_names:
        sub = df[df["fragment_name"] == frag].sort_values("rt")
        ax.plot(
            sub["rt"],
            sub["fragment_intensity"],
            color=color_map[frag],
            linewidth=1.2,
            alpha=0.9,
        )
        ax.scatter(
            sub["rt"], sub["fragment_intensity"], color=color_map[frag], s=10, alpha=0.9
        )

    if show_margins and {"rt_lower_margin", "rt_higher_margin"}.issubset(df.columns):
        margins = df[["rt_lower_margin", "rt_higher_margin"]].dropna()
        if not margins.empty:
            ax.axvline(
                float(margins["rt_lower_margin"].iloc[0]),
                color="gray",
                linestyle="--",
                linewidth=1,
            )
            ax.axvline(
                float(margins["rt_higher_margin"].iloc[0]),
                color="gray",
                linestyle="--",
                linewidth=1,
            )

    if x_limits is not None:
        ax.set_xlim(*x_limits)

    ax.set_xlabel("RT")
    ax.set_ylabel("Fragment intensity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_all_tables(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    debug_dir = Path(args.debug_dir)
    results_dir = Path(args.results_dir)
    comparison_dir = Path(args.comparison_dir)
    tables = {
        "stage2_psms": load_tsv(
            debug_dir / "df_psms_after_retention_window_searches.tsv"
        ),
        "stage2_fragments": ensure_fragment_name(
            load_tsv(debug_dir / "df_fragment_after_retention_window_searches.tsv")
        ),
        "post_rt_psms": load_tsv(debug_dir / "df_psms_after_rt.csv"),
        "post_ms2pip_psms": load_tsv(debug_dir / "df_psms_after_ms2pip.tsv"),
        "post_ms2pip_fragments": ensure_fragment_name(
            load_tsv(debug_dir / "df_fragment_after_ms2pip.tsv")
        ),
        "final_psms": load_tsv(results_dir / "df_psms.tsv"),
        "final_fragments": ensure_fragment_name(
            load_tsv(results_dir / "df_fragment.tsv")
        ),
        "mokapot_psms": load_tsv(results_dir / "mokapot.psms.txt"),
        "mokapot_peptides": load_tsv(results_dir / "mokapot.peptides.txt"),
        "merged_features": load_tsv(comparison_dir / "merged_features.tsv"),
        "gap_peptides": load_tsv(comparison_dir / "gap_peptides.tsv"),
    }
    return tables


def collect_comparison_rows(
    df: pd.DataFrame, peptide: str, charge: int
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "Peptide" in df.columns and "charge" in df.columns:
        return df[
            (df["Peptide"] == peptide) & (df["charge"].astype(int) == charge)
        ].copy()
    if "Peptide" in df.columns:
        return df[df["Peptide"] == peptide].copy()
    if "stripped_peptide" in df.columns:
        stripped = re.sub(r"\[.*?\]", "", peptide)
        mask = df["stripped_peptide"].astype(str) == stripped
        if "charge" in df.columns:
            mask &= df["charge"].astype(int) == charge
        return df[mask].copy()
    return df.iloc[0:0].copy()


def audit_one(
    peptidoform: str, tables: dict[str, pd.DataFrame], output_root: Path
) -> None:
    peptide, charge = parse_peptidoform(peptidoform)
    folder = output_root / slugify(f"{peptide}_{charge}")
    folder.mkdir(parents=True, exist_ok=True)

    stage2_psms = filter_peptidoform(tables["stage2_psms"], peptide, charge)
    stage2_fragments = filter_peptidoform(tables["stage2_fragments"], peptide, charge)
    post_rt_psms = filter_peptidoform(tables["post_rt_psms"], peptide, charge)
    post_ms2pip_psms = filter_peptidoform(tables["post_ms2pip_psms"], peptide, charge)
    post_ms2pip_fragments = filter_peptidoform(
        tables["post_ms2pip_fragments"], peptide, charge
    )
    final_psms = filter_peptidoform(tables["final_psms"], peptide, charge)
    final_fragments = filter_peptidoform(tables["final_fragments"], peptide, charge)
    mokapot_psms = filter_mokapot(tables["mokapot_psms"], peptide)
    mokapot_peptides = filter_mokapot(tables["mokapot_peptides"], peptide)
    merged_features = collect_comparison_rows(
        tables["merged_features"], peptide, charge
    )
    gap_peptides = collect_comparison_rows(tables["gap_peptides"], peptide, charge)

    write_tsv(stage2_psms, folder / "stage2_psms.tsv")
    write_tsv(stage2_fragments, folder / "stage2_fragments.tsv")
    write_tsv(post_rt_psms, folder / "post_rt_psms.tsv")
    write_tsv(post_ms2pip_psms, folder / "post_ms2pip_psms.tsv")
    write_tsv(post_ms2pip_fragments, folder / "post_ms2pip_fragments.tsv")
    write_tsv(final_psms, folder / "final_psms.tsv")
    write_tsv(final_fragments, folder / "final_fragments.tsv")
    write_tsv(mokapot_psms, folder / "mokapot_psms.tsv")
    write_tsv(mokapot_peptides, folder / "mokapot_peptides.tsv")
    write_tsv(merged_features, folder / "merged_features.tsv")
    write_tsv(gap_peptides, folder / "gap_peptides.tsv")

    stage2_limits = get_rt_limits(stage2_fragments)
    post_ms2pip_limits = get_margin_limits(post_ms2pip_fragments) or get_rt_limits(
        post_ms2pip_fragments
    )
    final_limits = get_rt_limits(final_fragments)

    plot_fragments(
        stage2_fragments,
        folder / "stage2_fragments.png",
        f"Stage 2 fragments: {peptide}/{charge}",
    )
    plot_fragments(
        stage2_fragments,
        folder / "stage2_fragments_zoom.png",
        f"Stage 2 fragments (zoom): {peptide}/{charge}",
        x_limits=stage2_limits,
    )
    plot_fragments(
        post_ms2pip_fragments,
        folder / "post_ms2pip_fragments.png",
        f"Post-MS2PIP fragments: {peptide}/{charge}",
        show_margins=True,
    )
    plot_fragments(
        post_ms2pip_fragments,
        folder / "post_ms2pip_fragments_zoom.png",
        f"Post-MS2PIP fragments (zoom): {peptide}/{charge}",
        show_margins=True,
        x_limits=post_ms2pip_limits,
    )
    plot_fragments(
        final_fragments,
        folder / "final_fragments.png",
        f"Final fragments: {peptide}/{charge}",
    )
    plot_fragments(
        final_fragments,
        folder / "final_fragments_zoom.png",
        f"Final fragments (zoom): {peptide}/{charge}",
        x_limits=final_limits,
    )

    stage2_fragment_names = {
        canonical_fragment_name(v)
        for v in stage2_fragments.get("fragment_name", pd.Series(dtype=str)).dropna()
    }
    post_ms2pip_fragment_names = {
        canonical_fragment_name(v)
        for v in post_ms2pip_fragments.get(
            "fragment_name", pd.Series(dtype=str)
        ).dropna()
    }
    final_fragment_names = {
        canonical_fragment_name(v)
        for v in final_fragments.get("fragment_name", pd.Series(dtype=str)).dropna()
    }

    summary = {
        "peptidoform": peptidoform,
        "normalized_peptide": peptide,
        "charge": charge,
        "stage2_psms": summarize_psms(stage2_psms),
        "stage2_fragments": summarize_fragments(stage2_fragments),
        "post_rt_psms": summarize_psms(post_rt_psms),
        "post_ms2pip_psms": summarize_psms(post_ms2pip_psms),
        "post_ms2pip_fragments": summarize_fragments(post_ms2pip_fragments),
        "final_psms": summarize_psms(final_psms),
        "final_fragments": summarize_fragments(final_fragments),
        "mokapot_psms_rows": int(len(mokapot_psms)),
        "mokapot_peptides_rows": int(len(mokapot_peptides)),
        "comparison_rows": int(len(merged_features)),
        "gap_rows": int(len(gap_peptides)),
        "lost_after_ms2pip": sorted(stage2_fragment_names - post_ms2pip_fragment_names),
        "gained_after_ms2pip": sorted(
            post_ms2pip_fragment_names - stage2_fragment_names
        ),
        "lost_after_final": sorted(post_ms2pip_fragment_names - final_fragment_names),
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        f"# Audit for {peptidoform}",
        "",
        "## Key files",
        "- `stage2_psms.tsv`: targeted-search PSMs immediately after stage 2",
        "- `stage2_fragments.tsv`: fragment rows immediately after stage 2",
        "- `post_rt_psms.tsv`: after RT prediction/error filtering",
        "- `post_ms2pip_psms.tsv`: after RT margins/count filter before correlation/XIC downstream use",
        "- `post_ms2pip_fragments.tsv`: fragment table with `fragment_name` and RT margins",
        "- `final_psms.tsv` / `final_fragments.tsv`: full-search saved outputs",
        "- `mokapot_*.tsv`: final rescored outputs",
        "- `merged_features.tsv`: MuMDIA vs DIA-NN comparison row if available",
        "",
        "## Suggested inspection order",
        "1. `stage2_fragments_zoom.png` and `stage2_fragments.tsv` — verify Sage targeted-search apex, RT spread, and whether expected fragment candidates are present right after stage 2.",
        "2. `post_rt_psms.tsv` — verify whether RT prediction filtering already removes or distorts the candidate.",
        "3. `post_ms2pip_fragments_zoom.png` and `post_ms2pip_fragments.tsv` — verify fragment naming, RT margins, and whether fragments disappear before XIC/correlation features are formed.",
        "4. `merged_features.tsv` and `mokapot_*.tsv` — only after the above, inspect whether the final q-value is a downstream consequence or a true scoring issue.",
        "",
        "## Fragment transitions",
        f"- Lost after MS2PIP/margin stage: {', '.join(summary['lost_after_ms2pip'][:20]) or 'none'}",
        f"- Gained after MS2PIP/margin stage: {', '.join(summary['gained_after_ms2pip'][:20]) or 'none'}",
        f"- Lost before final saved fragments: {', '.join(summary['lost_after_final'][:20]) or 'none'}",
        "",
        "## Snapshot counts",
        f"- Stage 2 PSM rows: {summary['stage2_psms'].get('n_rows', 0)}",
        f"- Stage 2 fragment rows: {summary['stage2_fragments'].get('n_rows', 0)}",
        f"- Post-RT PSM rows: {summary['post_rt_psms'].get('n_rows', 0)}",
        f"- Post-MS2PIP PSM rows: {summary['post_ms2pip_psms'].get('n_rows', 0)}",
        f"- Post-MS2PIP fragment rows: {summary['post_ms2pip_fragments'].get('n_rows', 0)}",
        f"- Final mokapot peptide rows: {summary['mokapot_peptides_rows']}",
    ]
    (folder / "README.md").write_text("\n".join(md_lines))


def main() -> None:
    args = parse_args()
    peptidoforms = list(args.peptidoforms or [])
    if args.peptidoform_file:
        peptidoforms.extend(load_peptidoforms_from_file(args.peptidoform_file))
    if not peptidoforms:
        peptidoforms = DEFAULT_PEPTIDOFORMS
    peptidoforms = list(dict.fromkeys(peptidoforms))
    tables = load_all_tables(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for peptidoform in peptidoforms:
        audit_one(peptidoform, tables, output_root)
    print(f"Wrote audit bundles for {len(peptidoforms)} peptidoforms to {output_root}")


if __name__ == "__main__":
    main()
