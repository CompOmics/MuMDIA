#!/usr/bin/env python3
"""Plot apex decisions for the audited peptidoforms across Stage 3 steps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

UNIMOD_REPLACEMENTS = {
    "[UniMod:4]": "[Carbamidomethyl]",
    "[UniMod:35]": "[Oxidation]",
}

RT_JOIN_DECIMALS = 5

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
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("--output-dir", default="audit_peptidoforms_20260405")
    parser.add_argument(
        "--stage2-path",
        help="Optional override for the Stage 2 fragment table.",
    )
    parser.add_argument(
        "--candidate-path",
        help="Optional override for the candidate-window fragment table.",
    )
    parser.add_argument(
        "--post-margin-path",
        help="Optional override for the post-margin fragment table.",
    )
    parser.add_argument(
        "--initial-apex-path",
        help="Optional override for the initial apex table.",
    )
    parser.add_argument(
        "--final-apex-path",
        help="Optional override for the final apex table.",
    )
    parser.add_argument(
        "--psms-path",
        help="Optional PSM table used to derive per-peptidoform RT support for top-trace plotting.",
    )
    parser.add_argument(
        "--global-ms2-rt-grid-path",
        help="Optional TSV with a single rt column used for zero-filled top-trace plotting.",
    )
    parser.add_argument(
        "--suspicious-tsv",
        help="Optional TSV of suspicious mirror-plot candidates with Peptide and charge columns.",
    )
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
    peptide, charge = str(text).strip().rsplit("/", 1)
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


# Cache for pre-loaded full tables (path → pl.DataFrame)
_TABLE_CACHE: dict[str, pl.DataFrame] = {}


def _load_full_table(path: Path) -> pl.DataFrame:
    """Load and cache an entire TSV so repeated per-peptidoform queries are fast."""
    key = str(path)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    if not path.exists():
        _TABLE_CACHE[key] = pl.DataFrame()
        return _TABLE_CACHE[key]
    try:
        df = pl.read_csv(str(path), separator="\t", infer_schema_length=10000)
    except Exception:
        df = pl.DataFrame()
    _TABLE_CACHE[key] = df
    return df


def load_filtered_table(path: Path, peptide: str, charge: int) -> pd.DataFrame:
    df = _load_full_table(path)
    if df.is_empty():
        return pd.DataFrame()
    if "peptide" not in df.columns or "charge" not in df.columns:
        return pd.DataFrame()
    filtered = df.filter((pl.col("peptide") == peptide) & (pl.col("charge") == charge))
    return filtered.to_pandas()


def load_apex_row(path: Path, peptide: str, charge: int) -> dict | None:
    df = _load_full_table(path)
    if df.is_empty():
        return None
    if "peptide" not in df.columns or "charge" not in df.columns:
        return None
    filtered = df.filter((pl.col("peptide") == peptide) & (pl.col("charge") == charge))
    if filtered.is_empty():
        return None
    return filtered.head(1).to_dicts()[0]


def load_rt_grid(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    df = pl.read_csv(path, separator="\t")
    if df.is_empty() or "rt" not in df.columns:
        return None
    return df["rt"].to_numpy().astype(np.float64)


def infer_fragment_source_label(path: Path | None, df: pd.DataFrame) -> str:
    """Best-effort label describing which fragment table a panel represents."""
    if path is not None:
        summary_path = path.parent / "margin_replay_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                fragment_file_used = str(summary.get("fragment_file_used", ""))
                if "reannotated" in fragment_file_used:
                    return "reannotated fragments"
                if "search" in fragment_file_used:
                    return "search fragments"
            except Exception:
                pass

        metadata_path = path.parent / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                fragment_file_used = str(
                    metadata.get("fragment_table_used_for_margin_calculation", "")
                )
                if "reannotated" in fragment_file_used:
                    return "reannotated fragments"
                if "search" in fragment_file_used:
                    return "search fragments"
            except Exception:
                pass

        lowered = path.name.lower()
        if "candidate_window" in lowered or "initial_search" in lowered:
            return "search fragments"
        if "after_ms2pip" in lowered:
            return "reannotated fragments"
        if "reannotated" in lowered:
            return "reannotated fragments"

    cols = set(df.columns)
    if {"fragment_mz_experimental", "fragment_mz_calculated"}.issubset(cols):
        return "search fragments"
    if "fragment_mz" in cols and "fragment_mz_experimental" not in cols:
        return "reannotated fragments"
    return "fragment table"


def infer_apex_source_label(path: Path | None) -> str:
    if path is None:
        return "apex table"
    lowered = path.name.lower()
    if "replayed" in lowered:
        return "replayed apex"
    if "after_ms2pip" in lowered:
        return "pipeline apex"
    return "apex table"


def filter_to_margin_window(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a fragment table to its calibrated RT window when margins exist."""
    if df.empty or not {"rt", "rt_lower_margin", "rt_higher_margin"}.issubset(
        df.columns
    ):
        return df.copy()

    margins = (
        df[["rt_lower_margin", "rt_higher_margin"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    if margins.empty:
        return df.copy()

    left = float(margins["rt_lower_margin"].iloc[0])
    right = float(margins["rt_higher_margin"].iloc[0])
    rt = pd.to_numeric(df["rt"], errors="coerce")
    return df.loc[rt.between(left, right, inclusive="both")].copy()


def load_peptidoforms_from_suspicious_tsv(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Suspicious TSV not found: {path}")

    df = pl.read_csv(path, separator="\t")
    required = {"Peptide", "charge"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Suspicious TSV must contain columns {sorted(required)}, got {df.columns}"
        )

    peptidoforms: list[str] = []
    seen: set[str] = set()
    for row in df.select(["Peptide", "charge"]).unique(maintain_order=True).to_dicts():
        peptide = str(row["Peptide"]) if row.get("Peptide") is not None else ""
        charge = int(float(row["charge"]))
        peptidoform = f"{peptide}/{charge}"
        if peptidoform not in seen:
            peptidoforms.append(peptidoform)
            seen.add(peptidoform)
    return peptidoforms


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
            + "/"
            + out["fragment_charge"].astype(int).astype(str)
        )
    return out


def get_xlim(*frames: pd.DataFrame) -> tuple[float, float] | None:
    rt_vals = []
    for df in frames:
        if not df.empty and "rt" in df.columns:
            vals = pd.to_numeric(df["rt"], errors="coerce").dropna().tolist()
            rt_vals.extend(vals)
    if not rt_vals:
        return None
    left = min(rt_vals)
    right = max(rt_vals)
    span = max(right - left, 0.05)
    pad = max(span * 0.08, 0.02)
    return left - pad, right + pad


def get_preferred_fragments(apex_row: dict | None, top_n: int = 2) -> list[str]:
    if not apex_row:
        return []
    fragments_raw = apex_row.get("top_predicted_fragments")
    if not fragments_raw:
        return []
    return [frag for frag in str(fragments_raw).split(";") if frag][:top_n]


def build_top_trace(
    df: pd.DataFrame,
    preferred_fragments: list[str],
    psm_df: pd.DataFrame | None = None,
    global_ms2_rt_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    if df.empty or not preferred_fragments:
        return pd.DataFrame(columns=["rt", "trace_intensity"])

    df = ensure_fragment_name(df)
    trace_source = df[df["fragment_name"].isin(preferred_fragments)].copy()
    if trace_source.empty:
        return pd.DataFrame(columns=["rt", "trace_intensity"])

    trace = (
        trace_source.groupby("rt", as_index=False)["fragment_intensity"]
        .sum()
        .rename(columns={"fragment_intensity": "trace_intensity"})
        .sort_values("rt")
    )

    if psm_df is None or psm_df.empty:
        return trace

    psm_rts = (
        pd.to_numeric(psm_df["rt"], errors="coerce").dropna().to_numpy(dtype=float)
    )
    if psm_rts.size == 0:
        return trace

    fill_rts = np.unique(psm_rts)
    if global_ms2_rt_grid is not None and global_ms2_rt_grid.size > 0:
        lower_rt = float(np.min(psm_rts))
        upper_rt = float(np.max(psm_rts))
        start_idx = int(np.searchsorted(global_ms2_rt_grid, lower_rt, side="left"))
        end_idx = int(np.searchsorted(global_ms2_rt_grid, upper_rt, side="right"))
        global_slice = global_ms2_rt_grid[start_idx:end_idx]
        if global_slice.size > 0:
            fill_rts = np.unique(np.concatenate([fill_rts, global_slice]))

    rt_frame = pd.DataFrame({"rt": np.sort(fill_rts)})
    rt_frame["_rt_key"] = rt_frame["rt"].round(RT_JOIN_DECIMALS)
    trace = trace.copy()
    trace["_rt_key"] = pd.to_numeric(trace["rt"], errors="coerce").round(
        RT_JOIN_DECIMALS
    )
    trace = (
        trace.groupby("_rt_key", as_index=False)["trace_intensity"]
        .sum()
        .sort_values("_rt_key")
    )
    merged = rt_frame.merge(trace, on="_rt_key", how="left")
    merged["trace_intensity"] = merged["trace_intensity"].fillna(0.0)
    return merged[["rt", "trace_intensity"]]


def build_fragment_trace(
    df: pd.DataFrame,
    fragment_name: str,
    psm_df: pd.DataFrame | None = None,
    global_ms2_rt_grid: np.ndarray | None = None,
    zero_fill: bool = False,
) -> pd.DataFrame:
    if df.empty or not fragment_name:
        return pd.DataFrame(columns=["rt", "trace_intensity"])

    df = ensure_fragment_name(df)
    frag_df = df[df["fragment_name"] == fragment_name].copy()
    if frag_df.empty:
        return pd.DataFrame(columns=["rt", "trace_intensity"])

    trace = (
        frag_df.groupby("rt", as_index=False)["fragment_intensity"]
        .sum()
        .rename(columns={"fragment_intensity": "trace_intensity"})
        .sort_values("rt")
    )

    if not zero_fill:
        return trace

    if psm_df is None or psm_df.empty:
        return trace

    psm_rts = (
        pd.to_numeric(psm_df["rt"], errors="coerce").dropna().to_numpy(dtype=float)
    )
    if psm_rts.size == 0:
        return trace

    fill_rts = np.unique(psm_rts)
    if global_ms2_rt_grid is not None and global_ms2_rt_grid.size > 0:
        lower_rt = float(np.min(psm_rts))
        upper_rt = float(np.max(psm_rts))
        start_idx = int(np.searchsorted(global_ms2_rt_grid, lower_rt, side="left"))
        end_idx = int(np.searchsorted(global_ms2_rt_grid, upper_rt, side="right"))
        global_slice = global_ms2_rt_grid[start_idx:end_idx]
        if global_slice.size > 0:
            fill_rts = np.unique(np.concatenate([fill_rts, global_slice]))

    rt_frame = pd.DataFrame({"rt": np.sort(fill_rts)})
    rt_frame["_rt_key"] = rt_frame["rt"].round(RT_JOIN_DECIMALS)
    trace = trace.copy()
    trace["_rt_key"] = pd.to_numeric(trace["rt"], errors="coerce").round(
        RT_JOIN_DECIMALS
    )
    trace = (
        trace.groupby("_rt_key", as_index=False)["trace_intensity"]
        .sum()
        .sort_values("_rt_key")
    )
    merged = rt_frame.merge(trace, on="_rt_key", how="left")
    merged["trace_intensity"] = merged["trace_intensity"].fillna(0.0)
    return merged[["rt", "trace_intensity"]]


def plot_panel(
    ax,
    df: pd.DataFrame,
    title: str,
    x_limits: tuple[float, float] | None,
    initial_apex: dict | None,
    final_apex: dict | None,
    top_trace: pd.DataFrame | None = None,
    top_fragment_traces: list[tuple[str, pd.DataFrame]] | None = None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("RT")
    ax.set_ylabel("Fragment intensity")

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if x_limits:
            ax.set_xlim(*x_limits)
        return

    df = ensure_fragment_name(df)
    fragment_names = sorted(df["fragment_name"].dropna().unique())
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(fragment_names), 1))

    for idx, frag in enumerate(fragment_names):
        sub = (
            df[df["fragment_name"] == frag]
            .groupby("rt", as_index=False)["fragment_intensity"]
            .sum()
            .sort_values("rt")
        )
        ax.plot(
            sub["rt"],
            sub["fragment_intensity"],
            color=cmap(idx),
            alpha=0.75,
            linewidth=1.0,
            label=frag,
        )

    if top_trace is not None and not top_trace.empty:
        ax.plot(
            top_trace["rt"],
            top_trace["trace_intensity"],
            color="black",
            linewidth=2.2,
            alpha=0.95,
            label="top-2 sum (aligned)",
        )

    fragment_styles = [
        ("darkorange", "--", "o"),
        ("deepskyblue", "-.", "s"),
    ]
    if top_fragment_traces:
        for idx, (fragment_name, fragment_trace) in enumerate(top_fragment_traces[:2]):
            if fragment_trace is None or fragment_trace.empty:
                continue
            color, linestyle, marker = fragment_styles[
                min(idx, len(fragment_styles) - 1)
            ]
            ax.plot(
                fragment_trace["rt"],
                fragment_trace["trace_intensity"],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                alpha=0.98,
                marker=marker,
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=0.8,
                zorder=5 + idx,
                label=f"{fragment_name} observed",
            )

    if initial_apex and initial_apex.get("rt") is not None:
        ax.axvline(
            float(initial_apex["rt"]),
            color="royalblue",
            linestyle="--",
            linewidth=1.8,
            label="initial apex",
        )
    if initial_apex and initial_apex.get("predicted_rt_anchor") is not None:
        ax.axvline(
            float(initial_apex["predicted_rt_anchor"]),
            color="royalblue",
            linestyle=":",
            linewidth=1.4,
            label="predicted RT",
        )
    if final_apex and final_apex.get("rt") is not None:
        ax.axvline(
            float(final_apex["rt"]),
            color="crimson",
            linestyle="--",
            linewidth=1.8,
            label="final apex",
        )

    if not df.empty and {"rt_lower_margin", "rt_higher_margin"}.issubset(df.columns):
        margins = (
            df[["rt_lower_margin", "rt_higher_margin"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        if not margins.empty:
            ax.axvline(
                float(margins["rt_lower_margin"].iloc[0]),
                color="gray",
                linestyle="-.",
                linewidth=1.2,
                label="lower margin",
            )
            ax.axvline(
                float(margins["rt_higher_margin"].iloc[0]),
                color="gray",
                linestyle="-.",
                linewidth=1.2,
                label="upper margin",
            )

    if x_limits:
        ax.set_xlim(*x_limits)

    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    if dedup:
        # Separate fragment ion labels from annotation lines so ions go in a
        # compact multi-column block and vertical-line annotations stay clean.
        ion_labels = {l: h for l, h in dedup.items() if re.match(r"^[by]\d+", l)}
        other_labels = {l: h for l, h in dedup.items() if l not in ion_labels}
        n_ion_cols = max(1, min(6, (len(ion_labels) + 3) // 4))
        if ion_labels and other_labels:
            leg1 = ax.legend(
                other_labels.values(),
                other_labels.keys(),
                fontsize=7.5,
                loc="upper right",
            )
            ax.add_artist(leg1)
            ax.legend(
                ion_labels.values(),
                ion_labels.keys(),
                fontsize=6.5,
                loc="upper left",
                ncols=n_ion_cols,
                handlelength=1.2,
                columnspacing=0.8,
                title="fragment ions",
                title_fontsize=7,
            )
        else:
            n_cols = max(1, min(6, (len(dedup) + 3) // 4))
            ax.legend(
                dedup.values(),
                dedup.keys(),
                fontsize=7,
                loc="best",
                ncols=n_cols,
                handlelength=1.2,
                columnspacing=0.8,
            )


def main() -> None:
    args = parse_args()
    debug_dir = Path(args.debug_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.suspicious_tsv:
        peptidoforms = load_peptidoforms_from_suspicious_tsv(Path(args.suspicious_tsv))
    else:
        peptidoforms = list(args.peptidoforms or [])
    if args.peptidoform_file:
        peptidoforms.extend(load_peptidoforms_from_file(args.peptidoform_file))
    if not peptidoforms:
        peptidoforms = DEFAULT_PEPTIDOFORMS
    peptidoforms = list(dict.fromkeys(peptidoforms))

    stage2_path = (
        Path(args.stage2_path)
        if args.stage2_path
        else debug_dir / "df_fragment_after_retention_window_searches.tsv"
    )
    candidate_path = (
        Path(args.candidate_path)
        if args.candidate_path
        else debug_dir / "df_fragment_candidate_window.tsv"
    )
    post_margin_path = (
        Path(args.post_margin_path)
        if args.post_margin_path
        else debug_dir / "df_fragment_after_ms2pip.tsv"
    )
    initial_apex_path = (
        Path(args.initial_apex_path)
        if args.initial_apex_path
        else debug_dir / "df_fragment_predicted_apex_initial.tsv"
    )
    final_apex_path = (
        Path(args.final_apex_path)
        if args.final_apex_path
        else debug_dir / "df_fragment_max_peptide_after_ms2pip.tsv"
    )
    psms_path = Path(args.psms_path) if args.psms_path else None
    if psms_path is None:
        replayed_psms_path = (
            post_margin_path.parent / "df_psms_with_replayed_margins.tsv"
        )
        if replayed_psms_path.exists():
            psms_path = replayed_psms_path
    global_ms2_rt_grid_path = (
        Path(args.global_ms2_rt_grid_path) if args.global_ms2_rt_grid_path else None
    )
    global_ms2_rt_grid = load_rt_grid(global_ms2_rt_grid_path)

    stage2_source_label = infer_fragment_source_label(stage2_path, pd.DataFrame())
    candidate_source_label = infer_fragment_source_label(candidate_path, pd.DataFrame())
    post_margin_source_label = infer_fragment_source_label(
        post_margin_path, pd.DataFrame()
    )
    final_apex_source_label = infer_apex_source_label(final_apex_path)
    if candidate_source_label != post_margin_source_label:
        print(
            "WARNING: candidate and post-margin panels use different fragment sources: "
            f"candidate={candidate_source_label}, post-margin={post_margin_source_label}."
        )
    if (
        final_apex_source_label == "replayed apex"
        and post_margin_source_label != "search fragments"
    ):
        print(
            "WARNING: final apex rows come from a replayed apex table while the post-margin panel uses "
            f"{post_margin_source_label}. Use debug/df_fragment_max_peptide_after_ms2pip.tsv for pipeline-consistent apex diagnostics."
        )

    for peptidoform in peptidoforms:
        peptide, charge = parse_peptidoform(peptidoform)
        folder = output_dir / slugify(f"{peptide}_{charge}")
        folder.mkdir(parents=True, exist_ok=True)

        stage2 = load_filtered_table(stage2_path, peptide, charge)
        candidate = load_filtered_table(candidate_path, peptide, charge)
        post_margin = load_filtered_table(post_margin_path, peptide, charge)
        post_margin_filtered = filter_to_margin_window(post_margin)
        psms = (
            load_filtered_table(psms_path, peptide, charge)
            if psms_path
            else pd.DataFrame()
        )
        initial_apex = load_apex_row(initial_apex_path, peptide, charge)
        final_apex = load_apex_row(final_apex_path, peptide, charge)
        preferred_fragments = get_preferred_fragments(final_apex, top_n=2)

        stage2_trace = build_top_trace(
            stage2, preferred_fragments, psms, global_ms2_rt_grid
        )
        candidate_trace = build_top_trace(
            candidate, preferred_fragments, psms, global_ms2_rt_grid
        )
        post_margin_trace = build_top_trace(
            post_margin, preferred_fragments, psms, global_ms2_rt_grid
        )
        post_margin_filtered_trace = build_top_trace(
            post_margin_filtered, preferred_fragments, psms, global_ms2_rt_grid
        )
        stage2_fragment_traces = [
            (
                fragment_name,
                build_fragment_trace(
                    stage2,
                    fragment_name,
                    psms,
                    global_ms2_rt_grid,
                    zero_fill=False,
                ),
            )
            for fragment_name in preferred_fragments
        ]
        candidate_fragment_traces = [
            (
                fragment_name,
                build_fragment_trace(
                    candidate,
                    fragment_name,
                    psms,
                    global_ms2_rt_grid,
                    zero_fill=False,
                ),
            )
            for fragment_name in preferred_fragments
        ]
        post_margin_fragment_traces = [
            (
                fragment_name,
                build_fragment_trace(
                    post_margin,
                    fragment_name,
                    psms,
                    global_ms2_rt_grid,
                    zero_fill=False,
                ),
            )
            for fragment_name in preferred_fragments
        ]
        post_margin_filtered_fragment_traces = [
            (
                fragment_name,
                build_fragment_trace(
                    post_margin_filtered,
                    fragment_name,
                    psms,
                    global_ms2_rt_grid,
                    zero_fill=False,
                ),
            )
            for fragment_name in preferred_fragments
        ]

        x_limits = get_xlim(stage2, candidate, post_margin, post_margin_filtered)

        fig, axes = plt.subplots(4, 1, figsize=(13, 18), sharex=True)
        plot_panel(
            axes[0],
            stage2,
            f"Stage 2 XIC ({stage2_source_label}) — {peptide}/{charge}",
            x_limits,
            initial_apex,
            final_apex,
            top_trace=stage2_trace,
            top_fragment_traces=stage2_fragment_traces,
        )
        plot_panel(
            axes[1],
            candidate,
            f"Candidate-window XIC ({candidate_source_label}) — {peptide}/{charge}",
            x_limits,
            initial_apex,
            final_apex,
            top_trace=candidate_trace,
            top_fragment_traces=candidate_fragment_traces,
        )
        plot_panel(
            axes[2],
            post_margin,
            f"Margin-annotated XIC ({post_margin_source_label}) — {peptide}/{charge}",
            x_limits,
            initial_apex,
            final_apex,
            top_trace=post_margin_trace,
            top_fragment_traces=post_margin_fragment_traces,
        )
        plot_panel(
            axes[3],
            post_margin_filtered,
            f"Margin-window filtered XIC ({post_margin_source_label}) — {peptide}/{charge}",
            x_limits,
            initial_apex,
            final_apex,
            top_trace=post_margin_filtered_trace,
            top_fragment_traces=post_margin_filtered_fragment_traces,
        )
        fig.tight_layout()
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / "apex_decision_xics.png", dpi=160)
        plt.close(fig)

    print(
        f"Wrote apex decision plots for {len(peptidoforms)} peptidoforms to {output_dir}"
    )


if __name__ == "__main__":
    main()
