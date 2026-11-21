"""Entrapment FDP estimates for an A/B on the E. coli + human entrapment library.

The E. coli AIF file is searched against an E. coli + 1:1 human entrapment library.
The human peptides cannot be in the vial, so any that pass at 1% are false by
construction: the empirical null that target-decoy q cannot provide.

It prints three estimates at peptide q 1% (`make_entrapment_fasta.fdp_estimates`): the
combined estimate and the lower bound for the whole accepted target list, and the estimate for
the real hits alone, `(ratio * entrapment + 1) / real`. The last is what
`rescore.classifier = entrapment` cuts on (`fdr::entrapment_q`), so on such a run it restates
the threshold; only the combined estimate and the lower bound say anything about calibration,
and only on a run thresholded on decoys.

Classification mirrors `rescore.rs::classify_entrapment`: a target row is a spike-in when
its protein contains the marker, does not contain the exclude string, and contains none of
the contaminant markers.

Usage:  python entrapment_fdp.py <config.json> <arm_label>:<out_dir> [<arm_label>:<out_dir> ...]
"""
import json
import sys

import pyarrow.parquet as pq

from make_entrapment_fasta import fdp_estimates

SCORED = "psms_scored.parquet"


def classify(protein, marker, exclude, contaminants):
    """(is_entrapment, is_real) for one protein string, as the engine classifies it."""
    ent = (
        marker is not None
        and marker in protein
        and (exclude is None or exclude not in protein)
        and not any(c in protein for c in contaminants)
    )
    return ent, not ent


def arm(label, out_dir, cfg):
    r = cfg.get("rescore", {})
    marker = r.get("entrapment_marker")
    exclude = r.get("entrapment_exclude")
    contaminants = r.get("entrapment_contaminant_markers", [])
    if "entrapment_ratio" not in r:
        raise SystemExit("the config has no rescore.entrapment_ratio; measure it with "
                         "make_entrapment_fasta.py --ratio-library and put it there")
    ratio = float(r["entrapment_ratio"])

    t = pq.read_table(
        f"{out_dir}/{SCORED}",
        columns=["label", "protein", "q_value", "peptide_q_value", "base_peptide_id"],
    ).to_pandas()

    is_decoy = t["label"].astype(str).str.lower().str.startswith("decoy")
    tgt = t[~is_decoy].copy()
    flags = tgt["protein"].map(lambda p: classify(str(p), marker, exclude, contaminants))
    tgt["is_ent"] = [a for a, _ in flags]

    # Peptide level, which is the unit docs/28 reports: one row per base peptide, the
    # group's own q as the engine wrote it.
    pep = tgt.sort_values("peptide_q_value").drop_duplicates("base_peptide_id")
    acc = pep[pep["peptide_q_value"] <= 0.01]
    n_real = int((~acc["is_ent"]).sum())
    n_ent = int(acc["is_ent"].sum())
    est = fdp_estimates(n_real, n_ent, ratio)

    # The decoy fraction is reported alongside because a changed FDP means nothing if
    # the decoy-based threshold moved at the same time.
    dec = t[is_decoy]
    n_dec_psm = int((dec["q_value"] <= 0.01).sum())
    n_tgt_psm = int((tgt["q_value"] <= 0.01).sum())

    return {
        "arm": label,
        "real peptides @1%": n_real,
        "spike-in peptides": n_ent,
        "combined FDP %": round(100 * est["combined"], 3),
        "lower bound %": round(100 * est["lower_bound"], 3),
        "real-only FDP %": round(100 * est["real_only"], 3),
        "target PSMs @1%": n_tgt_psm,
        "decoy PSMs @1%": n_dec_psm,
        "decoy fraction": round(n_dec_psm / max(n_tgt_psm, 1), 4),
        "entrapment_ratio": ratio,
    }


def main():
    cfg = json.load(open(sys.argv[1]))
    rows = []
    for spec in sys.argv[2:]:
        label, out_dir = spec.split(":", 1)
        try:
            rows.append(arm(label, out_dir, cfg))
        except Exception as exc:  # noqa: BLE001 - report, do not abort the other arm
            print(f"{label}: could not read ({exc})")
    if not rows:
        return 1
    keys = list(rows[0])
    width = max(len(k) for k in keys)
    for k in keys:
        print(f"  {k:<{width}} " + "".join(f"{str(r[k]):>18}" for r in rows))
    if len(rows) == 2:
        a, b = rows
        d = b["real peptides @1%"] - a["real peptides @1%"]
        pct = 100 * d / max(a["real peptides @1%"], 1)
        print()
        print(f"  peptides {d:+,} ({pct:+.2f}%) | combined FDP {a['combined FDP %']:.3f}% -> "
              f"{b['combined FDP %']:.3f}% | real-only {a['real-only FDP %']:.3f}% -> "
              f"{b['real-only FDP %']:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
