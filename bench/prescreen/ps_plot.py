"""Reduction-versus-retention figures from ps_agg.py summaries.

usage: ps_plot.py OUT.png NAME=summary.json [NAME=summary.json ...]
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1]
runs = [a.split("=", 1) for a in sys.argv[2:]]
LABEL = {"OFF": "off", "PRESCAN": "prescan (anchor_all)", "CAP300": "capped 300 (0.54)",
         "SEN": "sensitive 0.52", "BAL": "balanced 0.54", "STR": "stringent 0.75",
         "AGG": "aggressive 0.90", "X_FAM": "family support 0.54", "X_BEST": "best site 0.54",
         "X_OX53": "oxidation scope 0.53", "X_RET": "tag retrieval, no rescue"}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
marks = "osD^v<>p*hX"
for (name, path), mk in zip(runs, marks):
    s = json.load(open(path))["arms"]
    arms = [a for a in s if a in LABEL]
    red = [100 * (s[a]["reduction"] or 0) for a in arms]
    for ax, key, ylab in [(axes[0], "reference_retention", "accepted precursors of OFF kept (%)"),
                          (axes[1], "weak_retention", "weakest quartile of OFF kept (%)")]:
        ys = [100 * s[a][key] if s[a].get(key) is not None else None for a in arms]
        for x, y, a in zip(red, ys, arms):
            if y is None:
                continue
            ax.scatter(x, y, marker=mk, label=f"{name}" if a == arms[0] else None)
            ax.annotate(LABEL[a], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("candidates removed before extract (%)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    ys = [s[a].get("peptides_delta_pct") for a in arms]
    for x, y, a in zip(red, ys, arms):
        if y is None:
            continue
        axes[2].errorbar(x, y, yerr=100 * (s[a].get("peptides_sd") or 0) / (s["OFF"]["peptides_mean"] or 1),
                         marker=mk, linestyle="none", label=name if a == arms[0] else None)
        axes[2].annotate(LABEL[a], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_xlabel("candidates removed before extract (%)")
    axes[2].set_ylabel("peptides at 1% vs off (%, mean of NN seeds)")
    axes[2].grid(alpha=0.3)
for ax in axes:
    ax.legend(fontsize=8)
fig.suptitle("Prescreen: reduction versus retention and identifications")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(out)
