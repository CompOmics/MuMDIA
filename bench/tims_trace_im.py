#!/usr/bin/env python
"""Do the fragment traces jump between ions? (docs/TIMS_ROADMAP_bis.md, D2b)

  tims_trace_im.py --pops pops.parquet --run-dir output_mumdia_p6_best [--out d2b.parquet]

--pops is the (pop, candidate_id, peak_rank) table of tims_loss_features.py --save-pops
(R right-peak rejects, A accepted, Aq abundance-matched accepted, Dm score-matched decoys).
Per candidate, over its fragment traces (chromatograms v3, per-point `im`) inside the
scored peak bounds [elution_lo, elution_hi], and per trace with >= 1 non-zero point:
  n_grid      scan-grid points inside the bounds (peak width)
  n_pts       non-zero points in the peak (sparsity)
  im_sd       SD of the per-point 1/K0 (traces with >= 2 points)
  im_step     median |1/K0 step| between consecutive non-zero points
  off_frac    share of the trace's peak intensity carried by points more than --off from
              the candidate's apex 1/K0 (psms_extracted `apex_im`, or --ref im_pred_cal)
Candidate value = median over its observed traces; the table prints each population's median
and the AUC of R against Aq and Dm.

Reads the full chromatogram layout only (`extract.chromatogram_schema = 1`, which writes v3
on 4D data). The default trimmed layout (v4: `rt_axis`, `intensity_trimmed`, `im_trimmed`)
is refused with that instruction rather than misread.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from tims_loss_features import auc  # noqa: E402


def trace_stats(rt, inten, im, lo, hi, apex_im, off):
    g = (rt >= lo) & (rt <= hi)
    k = g & (inten > 0)
    n = int(k.sum())
    if n == 0:
        return None
    x, w = im[k], inten[k]
    far = np.abs(x - apex_im) > off if np.isfinite(apex_im) else np.zeros(n, bool)
    return (int(g.sum()), n, float(x.std()) if n >= 2 else np.nan,
            float(np.median(np.abs(np.diff(x)))) if n >= 2 else np.nan,
            float(w[far].sum() / w.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--off", type=float, default=0.01, help="1/K0 distance counted as another ion")
    ap.add_argument("--ref", default="apex_im", choices=["apex_im", "im_pred_cal"],
                    help="1/K0 that off_frac is measured from")
    ap.add_argument("--pad", type=float, default=0.0, help="widen the peak bounds by this many s")
    ap.add_argument("--out")
    a = ap.parse_args()
    r = a.run_dir.rstrip("/")
    pops = pd.read_parquet(a.pops)
    pops = pops[pops["peak_rank"] == 0]  # chromatograms belong to the selected apex row
    cids = pops["candidate_id"].unique()
    sc = pd.read_parquet(f"{r}/psms_scored.parquet", columns=["candidate_id", "elution_lo", "elution_hi"])
    ex = pd.read_parquet(f"{r}/psms_extracted.parquet", columns=["candidate_id", "peak_rank", a.ref])
    meta = sc[sc.candidate_id.isin(cids)].merge(ex[ex.peak_rank == 0], on="candidate_id") \
        .set_index("candidate_id")
    chrom = f"{r}/chromatograms.parquet"
    names = pq.ParquetFile(chrom).schema_arrow.names
    if "rt" not in names or "im" not in names:
        sys.exit(f"{chrom}: not a full-layout 4D table (columns {names}); rerun extract with "
                 "extract.chromatogram_schema = 1 (chromatograms v3)")
    ch = pq.read_table(chrom, columns=["candidate_id", "frag_name", "rt", "intensity", "im"],
                       filters=[("candidate_id", "in", cids.tolist())])
    ch = ch.filter(pc.invert(pc.starts_with(ch["frag_name"], "ms1")))
    rows = []
    cid_col = ch["candidate_id"].to_numpy()
    rts, ins, ims = (ch[c].to_pylist() for c in ("rt", "intensity", "im"))
    for c, rt, it, im in zip(cid_col, rts, ins, ims):
        if not rt:
            continue
        m = meta.loc[c]
        s = trace_stats(np.asarray(rt), np.asarray(it), np.asarray(im), m.elution_lo - a.pad, m.elution_hi + a.pad,
                        m[a.ref] if m[a.ref] is not None else np.nan, a.off)
        if s:
            rows.append((c, *s))
    t = pd.DataFrame(rows, columns=["candidate_id", "n_grid", "n_pts", "im_sd", "im_step", "off_frac"])
    per = t.groupby("candidate_id").median()
    per["n_traces"] = t.groupby("candidate_id").size()
    stats = ["n_traces", "n_grid", "n_pts", "im_sd", "im_step", "off_frac"]
    res = {}
    for p in ["R", "Aq", "Dm", "A"]:
        res[p] = pops[pops["pop"] == p][["candidate_id"]].merge(per, left_on="candidate_id", right_index=True)
    print(f"traces {len(t)}; candidates R {len(res['R'])} Aq {len(res['Aq'])} Dm {len(res['Dm'])} A {len(res['A'])}")
    tab = pd.DataFrame({f"med_{p}": [res[p][s].median() for s in stats] for p in res}, index=stats)
    tab["auc_R_Aq"] = [auc(res["R"][s].to_numpy(float), res["Aq"][s].to_numpy(float)) for s in stats]
    tab["auc_R_Dm"] = [auc(res["R"][s].to_numpy(float), res["Dm"][s].to_numpy(float)) for s in stats]
    print(tab.to_string(float_format=lambda x: f"{x:.4g}"))
    if a.out:
        pd.concat([v.assign(pop=k) for k, v in res.items()]).to_parquet(a.out, index=False)


if __name__ == "__main__":
    s = trace_stats(np.array([1.0, 2, 3]), np.array([0.0, 10, 30]), np.array([0, 0.80, 0.83]),
                    1, 3, 0.80, 0.01)
    assert s[:2] == (3, 2) and abs(s[4] - 0.75) < 1e-9
    main()
