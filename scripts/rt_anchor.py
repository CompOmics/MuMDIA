"""Build clean RT training anchors for DeepLC fine-tuning.

The seed search's observed_rt can sit on a bright interfering peak (the wrong-apex
problem that dominates GATE_LOSS). This pollutes the fine-tune training set. Here we
recompute, for each confident seed target peptide, a shape-based apex: take the peptide's
3 most intense predicted library fragments, sum their observed intensity across all MS2
scans whose isolation window covers the precursor, and take the RT of the maximum of that
summed extracted-ion chromatogram. That apex is a far cleaner observed RT for training.

Output: a seed-shaped parquet (peptidoform,label,spectrum_q,observed_rt=refined apex, +QC)
consumable by deeplc_finetune.py unchanged.

Usage: python rt_anchor.py <seed_psms> <lib_fragments> <spectra_ms2> <masscal_json> <out_parquet>
"""
import json
import sys
import numpy as np
import pandas as pd

TOP_N = 3


def main():
    seed_p, frag_p, ms2_p, masscal_p, out_p = sys.argv[1:6]
    with open(masscal_p) as fh:
        mc = json.load(fh)
    off_ppm = float(mc.get("frag_ppm_offset", 0.0))
    tol_ppm = float(mc.get("frag_tol_ppm", 20.0))

    seed = pd.read_parquet(seed_p, columns=["candidate_id", "peptidoform", "charge",
                                            "precursor_mz", "label", "spectrum_q", "observed_rt"])
    conf = seed[(seed.label == "target") & (seed.spectrum_q <= 0.01)].copy()
    conf = conf.drop_duplicates("candidate_id").reset_index(drop=True)
    cand_ids = set(conf.candidate_id)
    print(f"confident seed targets: {len(conf)}", flush=True)

    # top-N predicted fragments per confident candidate (m/z, offset-corrected)
    frag = pd.read_parquet(frag_p, columns=["candidate_id", "mz", "predicted_intensity"])
    frag = frag[frag.candidate_id.isin(cand_ids)]
    frag = frag.sort_values(["candidate_id", "predicted_intensity"], ascending=[True, False])
    top = frag.groupby("candidate_id").head(TOP_N)
    cand2frags = {c: g.mz.to_numpy() * (1.0 + off_ppm / 1e6) for c, g in top.groupby("candidate_id")}

    # Flat per-candidate fragment array padded to TOP_N (NaN = absent), in conf row order
    cid_arr = conf.candidate_id.to_numpy()
    F = np.full((len(conf), TOP_N), np.nan)
    for i, cid in enumerate(cid_arr):
        fm = cand2frags.get(cid)
        if fm is not None:
            F[i, :fm.size] = fm[:TOP_N]
    prec = conf.precursor_mz.to_numpy()

    ms2 = pd.read_parquet(ms2_p, columns=["rt_seconds", "window_lower", "window_upper", "mz", "intensity"])
    wl = ms2.window_lower.to_numpy(); wu = ms2.window_upper.to_numpy()
    rt = ms2.rt_seconds.to_numpy()
    mz_lists = ms2.mz.to_list(); int_lists = ms2.intensity.to_list()
    n_scan = len(ms2)

    apex_rt = np.full(len(conf), np.nan)
    apex_int = np.full(len(conf), -1.0)
    n_present = np.zeros(len(conf), dtype=int)

    # unique isolation windows -> scan indices
    uniq = {}
    for i in range(n_scan):
        uniq.setdefault((wl[i], wu[i]), []).append(i)
    print(f"{len(uniq)} unique isolation windows, {n_scan} MS2 scans", flush=True)

    for wk, (win, scans) in enumerate(uniq.items()):
        lo, hi = win
        cmask = np.where((prec >= lo) & (prec <= hi))[0]
        if cmask.size == 0:
            continue
        Fw = F[cmask]                       # (nc, TOP_N)
        nc = cmask.size
        qmz = Fw.reshape(-1)                # (nc*TOP_N,)
        valid = ~np.isnan(qmz)
        qv = np.where(valid, qmz, 0.0)
        tol = qv * tol_ppm / 1e6
        qlo = qv - tol; qhi = qv + tol
        best_sum = np.full(nc, -1.0)
        best_rt = np.full(nc, np.nan)
        best_np = np.zeros(nc, dtype=int)
        for si in scans:
            m = np.asarray(mz_lists[si], dtype=np.float64)
            if m.size == 0:
                continue
            it = np.asarray(int_lists[si], dtype=np.float64)
            if not np.all(np.diff(m) >= 0):
                o = np.argsort(m); m = m[o]; it = it[o]
            a = np.searchsorted(m, qlo); b = np.searchsorted(m, qhi)
            present = (b > a) & valid
            idx = np.clip(a, 0, m.size - 1)
            val = np.where(present, it[idx], 0.0)
            val = val.reshape(nc, TOP_N)
            ssum = val.sum(axis=1)
            spres = present.reshape(nc, TOP_N).sum(axis=1)
            upd = ssum > best_sum
            best_sum = np.where(upd, ssum, best_sum)
            best_rt = np.where(upd, rt[si], best_rt)
            best_np = np.where(upd, spres, best_np)
        # merge into globals (candidate may appear in overlapping windows -> keep max)
        for j, ci in enumerate(cmask):
            if best_sum[j] > apex_int[ci]:
                apex_int[ci] = best_sum[j]; apex_rt[ci] = best_rt[j]; n_present[ci] = best_np[j]
        if (wk + 1) % 20 == 0:
            print(f"  window {wk+1}/{len(uniq)}", flush=True)

    conf["observed_rt_seed"] = conf.observed_rt
    conf["observed_rt"] = apex_rt
    conf["apex_top3_intensity"] = np.maximum(apex_int, 0.0)
    conf["n_top3_present"] = n_present
    good = conf[conf.observed_rt.notna() & (conf.n_top3_present >= 2)].copy()
    print(f"anchors with >=2 of top-3 present: {len(good)} / {len(conf)}", flush=True)
    good.to_parquet(out_p, index=False)
    d = (good.observed_rt - good.observed_rt_seed).abs()
    print(f"|refined - seed| RT: median {d.median():.1f}s  p90 {d.quantile(.9):.1f}s  >60s: {100*(d>60).mean():.1f}%")
    print("wrote", out_p)


if __name__ == "__main__":
    main()
