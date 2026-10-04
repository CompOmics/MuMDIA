"""Reduction-versus-retention figures from ps_agg.py summaries, one row per dataset.

usage: ps_plot.py OUT.png NAME=summary.json [NAME=summary.json ...]
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1]
runs = [a.split("=", 1) for a in sys.argv[2:]]
PRESETS = ["OFF", "SEN", "BAL", "STR", "AGG"]
OTHER = {"PRESCAN": ("existing prescan (anchor_all)", "s", "tab:orange"),
         "CAP300": ("capped 300, target 0.54", "v", "tab:red"),
         "X_FAM": ("family support, 0.54", "P", "tab:green"),
         "X_BEST": ("best site, 0.54", "X", "tab:olive"),
         "X_OX53": ("oxidation scope, 0.53", "D", "tab:purple"),
         "X_RET": ("tag retrieval, no rescue", "^", "tab:brown")}
fig, axes = plt.subplots(len(runs), 3, figsize=(16, 4.6 * len(runs)), squeeze=False)
for row, (name, path) in enumerate(runs):
    s = json.load(open(path))["arms"]
    off = s["OFF"]["peptides_mean"]

    def pts(arm, key):
        r = s[arm]
        x = 100 * (r.get("reduction") or 0)
        if key == "pep":
            if r.get("peptides_delta_pct") is None:
                return None
            return x, r["peptides_delta_pct"], 100 * (r.get("peptides_sd") or 0) / off
        v = r.get(key)
        return None if v is None else (x, 100 * v, 0)

    for col, (key, ylab) in enumerate([("reference_retention", "OFF-accepted precursors kept (%)"),
                                       ("weak_retention", "weakest-quartile OFF precursors kept (%)"),
                                       ("pep", "peptides at 1% vs OFF (%, NN seed mean +/- sd)")]):
        ax = axes[row][col]
        p = [pts(a, key) for a in PRESETS if a in s]
        p = [q for q in p if q]
        ax.errorbar([q[0] for q in p], [q[1] for q in p], yerr=[q[2] for q in p], marker="o",
                    color="tab:blue", label="uncapped presets 0.52 / 0.54 / 0.75 / 0.90")
        for a, (lab, mk, colr) in OTHER.items():
            if a in s and (q := pts(a, key)):
                ax.errorbar([q[0]], [q[1]], yerr=[q[2]], marker=mk, color=colr, linestyle="none",
                            markersize=8, label=lab)
        if key == "pep":
            ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("library candidates removed before extract (%)")
        ax.set_ylabel(ylab)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=8, loc="lower left")
fig.suptitle("Fragment-rarity prescreen: reduction versus retention and identifications")
fig.tight_layout()
fig.savefig(out, dpi=120)
print(out)
