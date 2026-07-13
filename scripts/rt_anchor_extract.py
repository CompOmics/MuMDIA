"""Build clean RT training anchors for DeepLC fine-tuning, using the extract.rs apex rule.

Unlike rt_anchor.py (which takes the RT of the single scan with max summed top-3 fragment
intensity), this mirrors the Rust extractor's apex (extract.rs:441-464): over ALL predicted
fragments, group by scan, and among scans whose DISTINCT-matched-fragment count is within
`apex_count_tol` of the per-peptide maximum, take the scan maximizing the summed intensity of
its 3 most intense fragments. The count gate makes the apex robust to a lone bright interfering
fragment that would win a pure max-intensity apex on chimeric DIA.

Output: seed-shaped parquet (peptidoform,label,spectrum_q,observed_rt=apex, +QC), drop-in for
deeplc_finetune.py.

Usage: python rt_anchor_extract.py <seed_psms> <lib_fragments> <spectra_ms2> <masscal_json> <out_parquet>
       [--apex-count-tol 1] [--min-matched 2]
"""
import argparse
import json
import numpy as np
import pandas as pd

TOP_N = 3  # top fragments summed for the intensity tiebreak (matches extract.rs top-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_p"); ap.add_argument("frag_p"); ap.add_argument("ms2_p")
    ap.add_argument("masscal_p"); ap.add_argument("out_p")
    ap.add_argument("--apex-count-tol", type=int, default=1)   # extract.rs default
    ap.add_argument("--min-matched", type=int, default=2)      # keep anchor if apex has >= this many frags
    args = ap.parse_args()

    with open(args.masscal_p) as fh:
        mc = json.load(fh)
    off_ppm = float(mc.get("frag_ppm_offset", 0.0))
    tol_ppm = float(mc.get("frag_tol_ppm", 20.0))

    seed = pd.read_parquet(args.seed_p, columns=["candidate_id", "peptidoform", "charge",
                                                 "precursor_mz", "label", "spectrum_q", "observed_rt"])
    conf = seed[(seed.label == "target") & (seed.spectrum_q <= 0.01)].copy()
    conf = conf.drop_duplicates("candidate_id").reset_index(drop=True)
    cand_ids = set(conf.candidate_id)
    print(f"confident seed targets: {len(conf)}", flush=True)

    # ALL predicted fragments per confident candidate (offset-corrected m/z)
    frag = pd.read_parquet(args.frag_p, columns=["candidate_id", "mz", "predicted_intensity"])
    frag = frag[frag.candidate_id.isin(cand_ids)]
    frag = frag.sort_values(["candidate_id", "predicted_intensity"], ascending=[True, False])
    cand2mz = {c: g.mz.to_numpy() * (1.0 + off_ppm / 1e6) for c, g in frag.groupby("candidate_id")}
    MAXF = min(max((v.size for v in cand2mz.values()), default=0), 40)
    print(f"max predicted fragments/candidate used: {MAXF}", flush=True)

    cid_arr = conf.candidate_id.to_numpy()
    F = np.full((len(conf), MAXF), np.nan)
    for i, cid in enumerate(cid_arr):
        fm = cand2mz.get(cid)
        if fm is not None:
            F[i, :min(fm.size, MAXF)] = fm[:MAXF]
    prec = conf.precursor_mz.to_numpy()

    ms2 = pd.read_parquet(args.ms2_p, columns=["rt_seconds", "window_lower", "window_upper", "mz", "intensity"])
    wl = ms2.window_lower.to_numpy(); wu = ms2.window_upper.to_numpy(); rt = ms2.rt_seconds.to_numpy()
    mz_lists = ms2.mz.to_list(); int_lists = ms2.intensity.to_list()
    n_scan = len(ms2)

    apex_rt = np.full(len(conf), np.nan)
    apex_int = np.full(len(conf), -1.0)     # summed top-3 at apex (cross-window selection key)
    apex_np = np.zeros(len(conf), dtype=int)  # distinct frags present at apex

    uniq = {}
    for i in range(n_scan):
        uniq.setdefault((wl[i], wu[i]), []).append(i)
    print(f"{len(uniq)} unique isolation windows, {n_scan} MS2 scans", flush=True)

    for wk, (win, scans) in enumerate(uniq.items()):
        lo, hi = win
        cmask = np.where((prec >= lo) & (prec <= hi))[0]
        if cmask.size == 0:
            continue
        Fw = F[cmask]; nc = cmask.size
        qmz = Fw.reshape(-1); valid = ~np.isnan(qmz)
        qv = np.where(valid, qmz, 0.0); tol = qv * tol_ppm / 1e6
        qlo = qv - tol; qhi = qv + tol
        S = len(scans)
        Count = np.zeros((nc, S), dtype=np.int16)
        Top3 = np.zeros((nc, S), dtype=np.float64)
        rtw = np.empty(S)
        for k, si in enumerate(scans):
            rtw[k] = rt[si]
            m = np.asarray(mz_lists[si], dtype=np.float64)
            if m.size == 0:
                continue
            it = np.asarray(int_lists[si], dtype=np.float64)
            if not np.all(np.diff(m) >= 0):
                o = np.argsort(m); m = m[o]; it = it[o]
            a = np.searchsorted(m, qlo); b = np.searchsorted(m, qhi)
            present = (b > a) & valid
            idx = np.clip(a, 0, m.size - 1)
            val = np.where(present, it[idx], 0.0).reshape(nc, MAXF)
            Count[:, k] = present.reshape(nc, MAXF).sum(axis=1)
            # sum of 3 largest per row
            if MAXF > TOP_N:
                part = np.partition(val, MAXF - TOP_N, axis=1)[:, -TOP_N:]
                Top3[:, k] = part.sum(axis=1)
            else:
                Top3[:, k] = val.sum(axis=1)
        # per-candidate apex via extract.rs count gate, then top-3 intensity
        for j in range(nc):
            cnt = Count[j]; maxc = int(cnt.max())
            if maxc < 1:
                continue
            thresh = max(1, maxc - args.apex_count_tol)
            elig = cnt >= thresh
            if not elig.any():
                continue
            t3 = np.where(elig, Top3[j], -1.0)
            kbest = int(np.argmax(t3))
            ci = cmask[j]
            if t3[kbest] > apex_int[ci]:   # cross-window: keep the stronger apex
                apex_int[ci] = t3[kbest]; apex_rt[ci] = rtw[kbest]; apex_np[ci] = int(cnt[kbest])
        if (wk + 1) % 20 == 0:
            print(f"  window {wk+1}/{len(uniq)}", flush=True)

    conf["observed_rt_seed"] = conf.observed_rt
    conf["observed_rt"] = apex_rt
    conf["apex_top3_intensity"] = np.maximum(apex_int, 0.0)
    conf["n_matched_at_apex"] = apex_np
    good = conf[conf.observed_rt.notna() & (conf.n_matched_at_apex >= args.min_matched)].copy()
    print(f"anchors with >={args.min_matched} matched frags at apex: {len(good)} / {len(conf)}", flush=True)
    good.to_parquet(args.out_p, index=False)
    d = (good.observed_rt - good.observed_rt_seed).abs()
    print(f"|apex - seed| RT: median {d.median():.1f}s  p90 {d.quantile(.9):.1f}s  >60s: {100*(d>60).mean():.1f}%")
    print("wrote", args.out_p)


if __name__ == "__main__":
    main()
