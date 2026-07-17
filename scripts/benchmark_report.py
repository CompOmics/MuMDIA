#!/usr/bin/env python
"""Self-contained HTML benchmark report for MuMDIA sensitivity diagnostics.

Implements sensitivity_plan backlog P7.1 (spec 02 Section 8, spec 05). Assembles
the diagnostics this project already produces into a single, portable HTML file:

  * identification-loss waterfall (candidate audit metrics + stage counts);
  * candidate-audit stratification (rejection reason by charge / entrapment);
  * reference-apex top-K peak recall (self, and reference if present);
  * feature-family ablation (per model, most useful first);
  * empirical entrapment FDP (parsed from the holdout harness stdout).

Non-invasive: it only reads existing JSON / CSV / Parquet outputs and writes one
HTML file with all CSS inlined and every chart embedded as a base64 PNG data URI.
There are no external assets and no network access, so the report opens correctly
straight from disk. Any missing input renders as "not provided" for its section.

Deterministic: analysis and charts are reproducible for the same inputs. Only the
header timestamp varies (fix it with --stamp). Charts require matplotlib; if it is
unavailable the tables still render and the chart panels note the omission.

Interpreter: C:/Users/robbi/anaconda3/envs/py312_mumdia/python.exe
  (pyarrow, pandas, numpy, matplotlib).

Usage:
  python benchmark_report.py --out report.html
    [--audit-metrics candidate_audit.parquet.metrics.json]
    [--audit         candidate_audit.parquet]
    [--topk          topk.json]
    [--ablation      ablation_dir_or_csv]
    [--entrapment    entrapment_stdout.txt]
    [--title "..."] [--stamp "2026-07-17 12:00"]
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import html
import io
import json
import os
import re
import sys

# --------------------------------------------------------------------------- #
# matplotlib is optional at runtime: charts are omitted gracefully if absent.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover - defensive
    _HAVE_MPL = False

# Fixed PNG metadata so byte output does not carry a matplotlib version string or
# creation time, keeping identical inputs byte-reproducible across runs.
_PNG_META = {"Software": "mumdia-benchmark-report"}

# Canonical ordering of known rejection reasons, following the pipeline ladder
# (spec 02 Section 8). Unknown reasons are appended in first-seen order.
_REASON_ORDER = [
    "NOT_IN_SEARCH_SPACE",
    "NO_CANDIDATE",
    "NOT_GENERATED",
    "NO_FRAGMENT_TRACES",
    "NO_VALID_FRAGMENTS",
    "NO_PEAK_GROUP",
    "RT_PRUNED",
    "PEAK_NOT_SELECTED",
    "WRONG_VARIANT",
    "OUTCOMPETED_PEAK",
    "OUTCOMPETED_PEPTIDE",
    "OUTCOMPETED_PEPTIDOFORM",
    "OUTCOMPETED_LOCALIZATION",
    "OUTCOMPETED_DUPLICATE",
    "OUTCOMPETED_TARGET_DECOY",
    "FAILED_PRECURSOR_FDR",
    "FAILED_PEPTIDE_FDR",
    "FAILED_PROTEIN_FDR",
    "NOT_REPORTED",
    "REPORTED",
]
_REASON_RANK = {r: i for i, r in enumerate(_REASON_ORDER)}


# --------------------------------------------------------------------------- #
# small helpers
def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return esc(x)


def fmt_pct(x, digits: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return esc(x)


def fmt_num(x, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}g}"
    except (TypeError, ValueError):
        return esc(x)


def load_json(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def reason_sort_key(reason: str):
    return (_REASON_RANK.get(reason, len(_REASON_ORDER)), str(reason))


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white", metadata=_PNG_META)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def not_provided(msg: str = "not provided") -> str:
    return f'<p class="np">{esc(msg)}</p>'


def html_table(headers, rows, align_right_from: int = 1) -> str:
    """Render a responsive table. Columns from `align_right_from` are right-aligned."""
    th = "".join(
        f'<th class="{"num" if i >= align_right_from else ""}">{esc(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        tds = "".join(
            f'<td class="{"num" if i >= align_right_from else ""}">{c}</td>'
            for i, c in enumerate(row)
        )
        body.append(f"<tr>{tds}</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{th}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def img_panel(uri: str, caption: str = "") -> str:
    cap = f'<figcaption>{esc(caption)}</figcaption>' if caption else ""
    return f'<figure class="chart"><img alt="{esc(caption)}" src="{uri}">{cap}</figure>'


# --------------------------------------------------------------------------- #
# Section 2: identification-loss waterfall
def section_waterfall(metrics):
    if not metrics:
        return not_provided()

    wf = metrics.get("waterfall") or {}
    parts = []

    # stage-count funnel
    stage_rows = []
    for key, label in (
        ("search_space", "Search space (candidates)"),
        ("extracted", "Extracted"),
        ("competed", "Competed"),
        ("reported", "Reported"),
    ):
        if key in metrics and metrics[key] is not None:
            stage_rows.append((label, fmt_int(metrics[key])))
    if metrics.get("trace_recall") is not None:
        stage_rows.append(("Trace recall (extracted / search space)",
                           fmt_pct(metrics["trace_recall"], 3)))
    if metrics.get("q_threshold") is not None:
        stage_rows.append(("q threshold", fmt_num(metrics["q_threshold"])))
    if metrics.get("run_id"):
        stage_rows.append(("Run id", esc(metrics["run_id"])))
    if stage_rows:
        parts.append("<h3>Stage counts</h3>")
        parts.append(html_table(["Stage", "Value"], stage_rows))

    if not wf:
        parts.append("<h3>Waterfall</h3>")
        parts.append(not_provided("no waterfall block in metrics"))
        return "".join(parts)

    ordered = sorted(wf.items(), key=lambda kv: reason_sort_key(kv[0]))
    total = sum(v for _, v in ordered) or metrics.get("search_space") or 1

    # table
    wf_rows = [
        (esc(reason), fmt_int(count), fmt_pct(count / total, 2))
        for reason, count in ordered
    ]
    parts.append("<h3>Earliest-loss waterfall</h3>")
    parts.append(html_table(["Rejection reason", "Candidates", "% of total"], wf_rows))

    # chart (log scale: the earliest bucket dominates by orders of magnitude)
    if _HAVE_MPL:
        labels = [r for r, _ in ordered]
        counts = [c for _, c in ordered]
        fig, ax = plt.subplots(figsize=(8.2, max(2.2, 0.55 * len(labels) + 1.0)))
        ypos = list(range(len(labels)))[::-1]
        bars = ax.barh(ypos, counts, color="#3f6fb0", edgecolor="#26456f")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("candidates (log scale)")
        ax.set_title("Identification-loss waterfall")
        for rect, c in zip(bars, counts):
            ax.text(rect.get_width() * 1.05, rect.get_y() + rect.get_height() / 2,
                    f"{c:,}", va="center", ha="left", fontsize=8, color="#333")
        ax.margins(x=0.18)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        parts.append(img_panel(fig_to_data_uri(fig),
                               "Candidates lost at each stage (log-scaled)."))
    else:
        parts.append(not_provided("matplotlib unavailable: chart omitted"))

    return "".join(parts)


# --------------------------------------------------------------------------- #
# Section 3: candidate-audit stratification
def _crosstab(df, index_col, col_col, col_order=None, col_fmt=str):
    import pandas as pd

    ct = pd.crosstab(df[index_col], df[col_col])
    # order rows by the pipeline ladder
    ct = ct.reindex(sorted(ct.index, key=reason_sort_key))
    if col_order is not None:
        cols = [c for c in col_order if c in ct.columns]
        cols += [c for c in ct.columns if c not in cols]
    else:
        cols = sorted(ct.columns, key=lambda c: (str(type(c)), c))
    # reindex (not ct[cols]) so a list of booleans is read as labels, not a mask
    ct = ct.reindex(columns=cols)
    headers = ["Rejection reason"] + [col_fmt(c) for c in ct.columns] + ["Total"]
    rows = []
    for reason, series in ct.iterrows():
        vals = [fmt_int(v) for v in series.tolist()]
        rows.append([esc(reason)] + vals + [fmt_int(series.sum())])
    totals = ["<b>Total</b>"] + [f"<b>{fmt_int(ct[c].sum())}</b>" for c in ct.columns]
    totals += [f"<b>{fmt_int(ct.values.sum())}</b>"]
    rows.append(totals)
    return html_table(headers, rows)


def section_stratification(audit_path):
    if not audit_path or not os.path.isfile(audit_path):
        return not_provided()
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - defensive
        return not_provided(f"pyarrow unavailable: {exc}")

    schema = pq.read_schema(audit_path)
    want = [c for c in ("rejection_reason", "charge", "entrapment_label",
                        "target_decoy_label") if c in schema.names]
    if "rejection_reason" not in want:
        return not_provided("no rejection_reason column in audit parquet")

    df = pq.read_table(audit_path, columns=want).to_pandas()
    n = len(df)
    parts = [f'<p class="muted">{fmt_int(n)} candidate rows.</p>']

    if "charge" in df.columns:
        charges = sorted(c for c in df["charge"].dropna().unique())
        parts.append("<h3>Rejection reason by charge</h3>")
        parts.append(_crosstab(df, "rejection_reason", "charge",
                               col_order=charges, col_fmt=lambda c: f"charge {int(c)}"))

    if "entrapment_label" in df.columns:
        parts.append("<h3>Rejection reason by entrapment label</h3>")
        parts.append(_crosstab(df, "rejection_reason", "entrapment_label",
                               col_order=[False, True],
                               col_fmt=lambda c: f"entrapment={bool(c)}"))

    if "target_decoy_label" in df.columns:
        parts.append("<h3>Rejection reason by target / decoy</h3>")
        parts.append(_crosstab(df, "rejection_reason", "target_decoy_label",
                               col_order=["target", "decoy"],
                               col_fmt=str))

    return "".join(parts)


# --------------------------------------------------------------------------- #
# Section 4: top-K peak recall
_TOPK_KEYS = [
    ("frac_rank1", "Selected apex is peak rank-1"),
    ("frac_top3", "Selected apex in top-3"),
    ("frac_top5", "Selected apex in top-5"),
    ("frac_top10", "Selected apex in top-10"),
    ("frac_no_peak", "Selected apex in no enumerated peak"),
]


def section_topk(topk):
    if not topk:
        return not_provided()

    parts = []
    params = topk.get("params") or {}
    ctx_rows = []
    for key, label in (
        ("n_candidates_total", "Candidates total"),
        ("n_candidates_selected", "Candidates selected"),
        ("n_candidates_processed", "Candidates processed"),
        ("n_no_chrom_or_empty", "No chromatogram / empty"),
    ):
        if topk.get(key) is not None:
            ctx_rows.append((label, fmt_int(topk[key])))
    if params.get("rt_tol_s") is not None:
        ctx_rows.append(("RT tolerance (s)", fmt_num(params["rt_tol_s"])))
    if params.get("top_frags") is not None:
        ctx_rows.append(("Top fragments", fmt_int(params["top_frags"])))
    if ctx_rows:
        parts.append("<h3>Context</h3>")
        parts.append(html_table(["Item", "Value"], ctx_rows))

    self_m = topk.get("self")
    if not self_m:
        parts.append(not_provided("no 'self' block in topk json"))
        return "".join(parts)

    rows = []
    for key, label in _TOPK_KEYS:
        if key in self_m and self_m[key] is not None:
            rows.append((label, fmt_pct(self_m[key], 2)))
    for key, label in (
        ("mean_peaks_per_candidate", "Mean peaks / candidate"),
        ("median_peaks_per_candidate", "Median peaks / candidate"),
        ("frac_ge2_peaks", "Candidates with >= 2 peaks"),
        ("n_apex_matched_to_peak", "Apexes matched to a peak"),
        ("denominator", "Denominator"),
    ):
        if key in self_m and self_m[key] is not None:
            val = fmt_pct(self_m[key], 2) if key == "frac_ge2_peaks" else (
                fmt_num(self_m[key]) if "peaks_per" in key else fmt_int(self_m[key]))
            rows.append((label, val))
    parts.append("<h3>Self peak recall</h3>")
    parts.append(html_table(["Metric", "Value"], rows))

    # cumulative top-K bar chart
    if _HAVE_MPL:
        bar_keys = [("frac_rank1", "rank-1"), ("frac_top3", "top-3"),
                    ("frac_top5", "top-5"), ("frac_top10", "top-10")]
        xs = [lbl for k, lbl in bar_keys if self_m.get(k) is not None]
        ys = [self_m[k] for k, _ in bar_keys if self_m.get(k) is not None]
        if ys:
            fig, ax = plt.subplots(figsize=(6.0, 3.2))
            bars = ax.bar(xs, ys, color="#4c9a63", edgecolor="#2f6b40")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("fraction of candidates")
            ax.set_title("Selected apex within cumulative top-K")
            for rect, y in zip(bars, ys):
                ax.text(rect.get_x() + rect.get_width() / 2, y + 0.02,
                        f"{y * 100:.1f}%", ha="center", va="bottom", fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            parts.append(img_panel(fig_to_data_uri(fig),
                                   "Cumulative fraction where the selected apex is "
                                   "among the top-K enumerated peaks."))

    # reference (peak-oracle) block, if present
    ref = topk.get("reference")
    if isinstance(ref, dict) and ref:
        ref_rows = []
        for k in ("reference_apex_in_top_1", "reference_apex_in_top_3",
                  "reference_apex_in_top_5", "reference_apex_in_top_10"):
            if k in ref and ref[k] is not None:
                ref_rows.append((k.replace("_", " "), fmt_pct(ref[k], 2)))
        for k, v in ref.items():
            if not k.startswith("reference_apex_in_top_"):
                ref_rows.append((esc(k), fmt_num(v) if isinstance(v, float) else fmt_int(v)))
        parts.append("<h3>Reference-apex peak oracle</h3>")
        parts.append(html_table(["Metric", "Value"], ref_rows))
    else:
        parts.append('<p class="muted">Reference (DIA-NN) apex block not present; '
                     "self analysis only.</p>")

    return "".join(parts)


# --------------------------------------------------------------------------- #
# Section 5: feature-family ablation
def _resolve_ablation_csvs(path):
    if not path:
        return []
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return [path]
    if os.path.isdir(path):
        preferred = os.path.join(path, "feature_ablation.csv")
        found = []
        if os.path.isfile(preferred):
            found.append(preferred)
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if full != preferred and name.lower().endswith(".csv") and os.path.isfile(full):
                found.append(full)
        return found
    return []


def _to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def section_ablation(path):
    csvs = _resolve_ablation_csvs(path)
    if not csvs:
        if path:
            return not_provided(f"no ablation CSV found at {esc(path)}")
        return not_provided()

    rows = []
    for cpath in csvs:
        with open(cpath, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    if not rows:
        return not_provided("ablation CSV(s) contained no rows")

    parts = [f'<p class="muted">Source: {esc(", ".join(os.path.basename(c) for c in csvs))}</p>']

    models = []
    for r in rows:
        m = r.get("model") or "model"
        if m not in models:
            models.append(m)

    for model in models:
        mrows = [r for r in rows if (r.get("model") or "model") == model]
        # most useful first: most negative delta_vs_full (removal hurts most)
        mrows.sort(key=lambda r: (_to_int(r.get("delta_vs_full")) is None,
                                  _to_int(r.get("delta_vs_full")) if
                                  _to_int(r.get("delta_vs_full")) is not None else 0))
        headers = ["Feature family", "Baseline IDs", "New IDs",
                   "delta vs full", "Recommendation"]
        trows = []
        for r in mrows:
            delta = _to_int(r.get("delta_vs_full"))
            delta_s = f"{delta:+,}" if delta is not None else esc(r.get("delta_vs_full"))
            rec = r.get("recommendation") or ""
            rec_cls = {
                "KEEP": "rec-keep", "HARMFUL": "rec-harm",
                "REDUNDANT_BUT_INFORMATIVE": "rec-info", "REDUNDANT": "rec-red",
            }.get(rec, "")
            rec_html = f'<span class="rec {rec_cls}">{esc(rec)}</span>' if rec else ""
            trows.append([
                esc(r.get("feature_family")),
                fmt_int(r.get("baseline_identifications")),
                fmt_int(r.get("new_identifications")),
                delta_s,
                rec_html,
            ])
        parts.append(f"<h3>Model: {esc(model)}</h3>")
        parts.append(html_table(headers, trows))
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Section 6: empirical entrapment FDP
def section_entrapment(path):
    if not path or not os.path.isfile(path):
        return not_provided()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    rows = []
    m = re.search(r"E\.?coli targets\s*=\s*(\d+)", text)
    if m:
        rows.append(("E. coli targets", fmt_int(m.group(1))))
    m = re.search(r"human train-neg\s*=\s*(\d+)", text)
    if m:
        rows.append(("Human train-negatives", fmt_int(m.group(1))))
    m = re.search(r"test-null\s*=\s*(\d+)", text)
    if m:
        rows.append(("Held-out test null", fmt_int(m.group(1))))
    m = re.search(r"ratio\s*=\s*([\d.]+)", text)
    if m:
        rows.append(("Library-size ratio", fmt_num(m.group(1))))
    for m in re.finditer(
        r"held-out E\.?coli stripped seqs @\s*(\d+)%[^:]*:\s*(\d+)", text
    ):
        rows.append((f"Held-out E. coli stripped seqs @ {m.group(1)}%",
                     fmt_int(m.group(2))))
    m = re.search(
        r"at shipped q<=\s*(\d+)%:\s*E\.?coli\s*=\s*(\d+),\s*true FDR"
        r"[^=]*=\s*([\d.]+)%",
        text,
    )
    if m:
        rows.append((f"E. coli at shipped q <= {m.group(1)}%", fmt_int(m.group(2))))
        rows.append((f"True FDR on held-out null (q <= {m.group(1)}%)",
                     f"{m.group(3)}%"))

    parts = []
    if rows:
        parts.append(html_table(["Metric", "Value"], rows))
    else:
        parts.append(not_provided("no recognizable entrapment metrics in the file"))
    parts.append("<details><summary>Raw entrapment output</summary>"
                 f"<pre>{esc(text.strip())}</pre></details>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Section 7: limitations footer
def section_limitations(rendered):
    absent = [name for name, ok in rendered.items() if not ok]
    items = [
        "Diagnostics are computed on a single dataset (the E. coli / HYE example); "
        "no family, threshold, or component decision should be made from one dataset "
        "or one favourable subset (spec 03 Section 9, spec 05 Section 7).",
        "The identification-loss waterfall collapses all extraction losses to "
        "NO_PEAK_GROUP at artifact resolution; the in-extract audit sidecar is needed "
        "to separate NO_FRAGMENT_TRACES / NO_VALID_FRAGMENTS / PEAK_NOT_SELECTED / "
        "RT_PRUNED.",
        "Top-K self recall measures peak-selection opportunity only; the reference "
        "(DIA-NN) peak-oracle metric requires a reference report and is shown only "
        "when a reference block is present.",
        "Feature-family ablation is cross-validated on one dataset; a gain that flips "
        "sign across models or datasets is not a keep. Rerun with both models and a "
        "second dataset before acting.",
        "The entrapment held-out FDP is the accept / reject gate; a sensitivity gain "
        "that inflates empirical FDP must not be retained.",
    ]
    parts = ["<ul>"]
    for it in items:
        parts.append(f"<li>{esc(it)}</li>")
    parts.append("</ul>")
    if absent:
        pretty = ", ".join(esc(a) for a in absent)
        parts.append(f'<p class="muted">Sections not rendered (input absent): '
                     f"{pretty}.</p>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 0 4rem 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
               Arial, sans-serif;
  line-height: 1.5; color: #1c2530; background: #f4f6f9;
}
.container { max-width: 1040px; margin: 0 auto; padding: 1.5rem; }
header.report {
  background: #26456f; color: #fff; padding: 1.6rem 1.5rem; border-radius: 0 0 10px 10px;
}
header.report h1 { margin: 0 0 .3rem 0; font-size: 1.55rem; }
header.report .stamp { opacity: .85; font-size: .9rem; }
section {
  background: #fff; border: 1px solid #e2e7ee; border-radius: 10px;
  padding: 1.2rem 1.3rem; margin: 1.2rem 0; box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
section > h2 {
  margin: 0 0 .4rem 0; font-size: 1.2rem; color: #26456f;
  border-bottom: 2px solid #eef1f5; padding-bottom: .4rem;
}
section > h2 .sec-no {
  display: inline-block; min-width: 1.6rem; color: #7a8aa0; font-weight: 600;
}
h3 { font-size: 1rem; margin: 1rem 0 .4rem 0; color: #33445c; }
p.desc { margin: .2rem 0 .8rem 0; color: #55627a; font-size: .92rem; }
p.np {
  color: #8a94a6; font-style: italic; background: #f7f8fb; border: 1px dashed #d7dde7;
  padding: .5rem .7rem; border-radius: 6px; display: inline-block;
}
p.muted { color: #6b7688; font-size: .88rem; }
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: .3rem 0 .6rem; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td {
  text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #eef1f5;
  white-space: nowrap;
}
th { background: #f0f3f8; color: #33445c; font-weight: 600; position: sticky; top: 0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:hover { background: #f7f9fc; }
figure.chart { margin: .8rem 0; text-align: center; }
figure.chart img {
  max-width: 100%; height: auto; border: 1px solid #e2e7ee; border-radius: 8px;
  background: #fff;
}
figcaption { color: #6b7688; font-size: .82rem; margin-top: .35rem; }
.toc { font-size: .92rem; }
.toc a { color: #26456f; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.rec { font-size: .78rem; padding: .1rem .45rem; border-radius: 10px; font-weight: 600; }
.rec-keep { background: #dff3e4; color: #1f6b39; }
.rec-harm { background: #fde3e0; color: #9c2b20; }
.rec-info { background: #e5eefb; color: #26456f; }
.rec-red  { background: #eef0f4; color: #66707f; }
details { margin-top: .6rem; }
summary { cursor: pointer; color: #26456f; font-weight: 600; }
pre {
  overflow-x: auto; background: #0f1826; color: #d7e0ee; padding: .8rem;
  border-radius: 8px; font-size: .82rem; line-height: 1.45;
}
footer.report { color: #6b7688; font-size: .8rem; text-align: center; padding: 1rem; }
@media (prefers-color-scheme: dark) {
  body { background: #10151d; color: #d6dde8; }
  section { background: #1a212c; border-color: #2a3341; box-shadow: none; }
  section > h2 { color: #9db9e6; border-bottom-color: #2a3341; }
  h3 { color: #c3cede; }
  th { background: #232c39; color: #c3cede; }
  th, td { border-bottom-color: #2a3341; }
  tbody tr:hover { background: #212a37; }
  p.np { background: #202836; border-color: #2f3a49; color: #8a94a6; }
  p.desc, p.muted, figcaption { color: #97a3b6; }
  figure.chart img { border-color: #2a3341; }
}
"""


def build_html(title, stamp, sections):
    """sections: list of (num, anchor, name, description, body_html)."""
    toc = " · ".join(
        f'<a href="#{anchor}">{num}. {esc(name)}</a>'
        for num, anchor, name, _desc, _body in sections
    )
    blocks = []
    for num, anchor, name, desc, body in sections:
        desc_html = f'<p class="desc">{esc(desc)}</p>' if desc else ""
        blocks.append(
            f'<section id="{anchor}">'
            f'<h2><span class="sec-no">{num}.</span>{esc(name)}</h2>'
            f"{desc_html}{body}</section>"
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f'<header class="report"><h1>{esc(title)}</h1>'
        f'<div class="stamp">Generated {esc(stamp)}</div></header>'
        f'<div class="container"><nav class="toc">{toc}</nav>'
        f'{"".join(blocks)}'
        '<footer class="report">MuMDIA sensitivity benchmark report. '
        "All diagnostics are read-only over existing artifacts.</footer>"
        "</div></body></html>"
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output HTML file")
    ap.add_argument("--audit-metrics", default=None,
                    help="candidate_audit.parquet.metrics.json (waterfall)")
    ap.add_argument("--audit", default=None,
                    help="candidate_audit.parquet (stratification)")
    ap.add_argument("--topk", default=None, help="topk.json (peak recall)")
    ap.add_argument("--ablation", default=None,
                    help="feature-family ablation CSV or directory")
    ap.add_argument("--entrapment", default=None,
                    help="entrapment holdout stdout text file")
    ap.add_argument("--title", default="MuMDIA Sensitivity Benchmark Report")
    ap.add_argument("--stamp", default=None,
                    help="fixed header timestamp (default: current local time)")
    args = ap.parse_args()

    stamp = args.stamp or _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics = load_json(args.audit_metrics)
    topk = load_json(args.topk)

    # Section 1 body: run metadata + input inventory
    inputs = [
        ("Audit metrics (JSON)", args.audit_metrics),
        ("Candidate audit (Parquet)", args.audit),
        ("Top-K peak recall (JSON)", args.topk),
        ("Feature ablation (CSV/dir)", args.ablation),
        ("Entrapment stdout (text)", args.entrapment),
    ]
    meta_rows = []
    if metrics and metrics.get("run_id"):
        meta_rows.append(("Run id", esc(metrics["run_id"])))
    meta_rows.append(("Report generated", esc(stamp)))
    for label, path in inputs:
        present = bool(path) and os.path.exists(path)
        status = "present" if present else ("missing" if path else "not provided")
        shown = esc(path) if path else "&mdash;"
        meta_rows.append((label, f"{shown} <span class='muted'>({status})</span>"))
    header_body = html_table(["Item", "Value"], meta_rows, align_right_from=99)

    # build sections
    wf_body = section_waterfall(metrics)
    strat_body = section_stratification(args.audit)
    topk_body = section_topk(topk)
    abl_body = section_ablation(args.ablation)
    ent_body = section_entrapment(args.entrapment)

    rendered = {
        "Identification-loss waterfall": bool(metrics),
        "Candidate-audit stratification": bool(args.audit and os.path.isfile(args.audit)),
        "Top-K peak recall": bool(topk),
        "Feature-family ablation": bool(_resolve_ablation_csvs(args.ablation)),
        "Empirical entrapment FDP": bool(args.entrapment and os.path.isfile(args.entrapment)),
    }
    lim_body = section_limitations(rendered)

    sections = [
        (1, "run", "Run metadata",
         "Report inputs and provenance. Each downstream section renders only when "
         "its input is provided.", header_body),
        (2, "waterfall", "Identification-loss waterfall",
         "Where DIA-NN-only precursors are lost, from candidate audit metrics "
         "(spec 02 Section 8).", wf_body),
        (3, "stratification", "Candidate-audit stratification",
         "Earliest rejection reason grouped by charge and entrapment label.",
         strat_body),
        (4, "topk", "Top-K peak recall",
         "How often the selected apex is the strongest peak, and (with a reference) "
         "whether the reference apex is within the top-K peaks.", topk_body),
        (5, "ablation", "Feature-family ablation",
         "Cross-validated contribution of each feature family, most useful first "
         "(largest identification drop when removed).", abl_body),
        (6, "entrapment", "Empirical entrapment FDP",
         "Held-out entrapment identification count at a genuinely controlled FDP "
         "(the accept / reject gate).", ent_body),
        (7, "limitations", "Limitations",
         "What is single-dataset or not yet measured.", lim_body),
    ]

    doc = build_html(args.title, stamp, sections)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(doc)

    n_rendered = sum(1 for v in rendered.values() if v)
    print(f"[written] {args.out} ({os.path.getsize(args.out):,} bytes)")
    print(f"[sections] {n_rendered + 2}/7 rendered "
          f"(run metadata + limitations always render)")
    for name, ok in rendered.items():
        print(f"    {'rendered ' if ok else 'not-prov '} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
