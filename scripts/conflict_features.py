"""Cross-candidate interference and ambiguity features (non-invasive pass).

Implements sensitivity_plan backlog items P2.1 (fragment claimant index),
P2.2 (peak-group conflict graph), P2.3 (conflict features) and P5.5 (candidate
ambiguity) as a single pass over existing MuMDIA artifacts. It reads the
per-candidate PSM table and the per-transition chromatogram table, builds a
bounded fragment claimant index, and writes one row of conflict features per
candidate to `conflict.parquet`, joinable into rescoring by `candidate_id`.

The engine is not modified. This is a first-pass ("fix later") implementation:
correctness, bounded memory, and a working smoke test are prioritised over
completeness. All computation is deterministic (stable sorts, sorted iteration
where floats are summed; no RNG is used).

Two candidates "claim" the same fragment when
  * their fragment m/z agree within `--frag-tol-ppm`, AND
  * their candidate apex RT agree within `--rt-window-s`, AND
  * their precursor m/z agree within `--mz-precursor-tol` (co-isolatable).

Only the scalar chromatogram columns (candidate_id, frag_mz,
predicted_intensity) are read; the large list columns (rt, intensity) are never
loaded, which keeps memory bounded.

House style: literal prose, no em-dashes, no hardcoded secrets.
"""

import argparse
import math
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_psms(path, max_candidates):
    """Load per-candidate metadata. Returns a dict of numpy arrays keyed by a
    compact 0..C-1 candidate index, plus the sorted original candidate ids."""
    cols = ["candidate_id", "apex_rt", "precursor_mz", "charge", "label",
            "base_peptide_id"]
    t = pq.read_table(path, columns=cols)
    cid = t.column("candidate_id").to_numpy()
    # Stable sort by candidate_id for reproducible compact indexing.
    order = np.argsort(cid, kind="stable")
    cid = cid[order]
    apex_rt = t.column("apex_rt").to_numpy()[order]
    prec_mz = t.column("precursor_mz").to_numpy()[order]
    charge = t.column("charge").to_numpy()[order]
    label = np.asarray(t.column("label").to_pylist(), dtype=object)[order]
    base_pep = t.column("base_peptide_id").to_numpy()[order]

    if max_candidates is not None and max_candidates < len(cid):
        cid = cid[:max_candidates]
        apex_rt = apex_rt[:max_candidates]
        prec_mz = prec_mz[:max_candidates]
        charge = charge[:max_candidates]
        label = label[:max_candidates]
        base_pep = base_pep[:max_candidates]

    is_decoy = np.array([str(x).lower() == "decoy" for x in label], dtype=bool)
    return {
        "uids": cid,               # sorted original candidate ids (== compact order)
        "apex_rt": apex_rt.astype(np.float64),
        "prec_mz": prec_mz.astype(np.float64),
        "charge": charge.astype(np.int32),
        "is_decoy": is_decoy,
        "base_pep": base_pep.astype(np.int64),
    }


def load_prelim(path, uids):
    """Load prelim_score from the comp table, aligned to the compact index."""
    t = pq.read_table(path, columns=["candidate_id", "prelim_score"])
    cid = t.column("candidate_id").to_numpy()
    score = t.column("prelim_score").to_numpy()
    prelim = np.full(len(uids), np.nan, dtype=np.float64)
    comp = np.searchsorted(uids, cid)
    inb = comp < len(uids)
    ok = np.zeros(len(cid), dtype=bool)
    ok[inb] = uids[comp[inb]] == cid[inb]
    prelim[comp[ok]] = score[ok]
    return prelim


def load_fragments(path, uids, batch_size=1_000_000):
    """Stream the scalar chromatogram columns and map each transition to the
    compact candidate index. Fragments of candidates outside `uids` are dropped
    (this happens when --max-candidates subsets the candidate set)."""
    pf = pq.ParquetFile(path)
    cols = ["candidate_id", "frag_mz", "predicted_intensity"]
    comp_chunks, mz_chunks, pint_chunks = [], [], []
    for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
        cid = batch.column("candidate_id").to_numpy(zero_copy_only=False)
        mz = batch.column("frag_mz").to_numpy(zero_copy_only=False)
        pint = batch.column("predicted_intensity").to_numpy(zero_copy_only=False)
        comp = np.searchsorted(uids, cid)
        inb = comp < len(uids)
        keep = np.zeros(len(cid), dtype=bool)
        keep[inb] = uids[comp[inb]] == cid[inb]
        if not keep.any():
            continue
        comp_chunks.append(comp[keep].astype(np.int32))
        mz_chunks.append(mz[keep].astype(np.float64))
        pint_chunks.append(pint[keep].astype(np.float64))
    if not comp_chunks:
        empty_i = np.zeros(0, dtype=np.int32)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_f, empty_f
    return (np.concatenate(comp_chunks),
            np.concatenate(mz_chunks),
            np.concatenate(pint_chunks))


def build_bin_index(frag_bin, frag_rt):
    """Group fragments by m/z bin, sorted by candidate apex RT within each bin.
    Returns the sort order and a dict bin_value -> (start, end) into the sorted
    arrays. Within [start, end) the RT array is ascending, so RT-window
    neighbours are a range scan rather than a full scan."""
    order = np.lexsort((frag_rt, frag_bin))  # primary bin, secondary rt
    sbin = frag_bin[order]
    bin_index = {}
    if len(sbin):
        uniq, starts = np.unique(sbin, return_index=True)
        ends = np.append(starts[1:], len(sbin))
        for b, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
            bin_index[b] = (s, e)
    return order, bin_index


def softmax_entropy(scores):
    """Shannon entropy (nats) of the softmax over `scores`. Deterministic given
    a fixed input order. Returns 0.0 for a single element."""
    if len(scores) <= 1:
        return 0.0
    s = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        s = np.nan_to_num(s, nan=np.nanmin(s) if np.any(np.isfinite(s)) else 0.0)
    z = s - s.max()
    w = np.exp(z)
    tot = w.sum()
    if tot <= 0:
        return 0.0
    w = w / tot
    nz = w > 0
    return float(-np.sum(w[nz] * np.log(w[nz])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psms", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--comp", default=None,
                    help="optional comp.parquet with prelim_score for ambiguity features")
    ap.add_argument("--out", default="conflict.parquet")
    ap.add_argument("--frag-tol-ppm", type=float, default=20.0)
    ap.add_argument("--rt-window-s", type=float, default=30.0)
    ap.add_argument("--mz-precursor-tol", type=float, default=0.5)
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="cap the candidate set (both queries and claimant "
                         "universe); for fast smoke tests over the full chrom")
    args = ap.parse_args()

    t0 = time.time()
    log(f"[conflict] loading psms {args.psms}")
    meta = load_psms(args.psms, args.max_candidates)
    uids = meta["uids"]
    C = len(uids)
    log(f"[conflict] {C} candidates")

    prelim = None
    if args.comp:
        log(f"[conflict] loading prelim_score {args.comp}")
        prelim = load_prelim(args.comp, uids)

    log(f"[conflict] streaming fragments {args.chrom}")
    fc, fmz, fpint = load_fragments(args.chrom, uids)
    nfrag = len(fc)
    log(f"[conflict] {nfrag} transitions retained")

    # Per-fragment candidate-level RT and precursor m/z (co-isolation keys).
    frt = meta["apex_rt"][fc]
    fpmz = meta["prec_mz"][fc]

    # ppm-scaled log bins. Two m/z within frag_tol_ppm differ by ~one bin, so
    # queries also scan neighbour bins (b-1, b+1).
    log_step = math.log1p(args.frag_tol_ppm * 1e-6)
    fbin = np.floor(np.log(fmz) / log_step).astype(np.int64)

    order, bin_index = build_bin_index(fbin, frt)
    sbin = fbin[order]
    srt = frt[order]
    smz = fmz[order]
    spmz = fpmz[order]
    scid = fc[order]
    spint = fpint[order]

    ppm = args.frag_tol_ppm * 1e-6
    rtw = args.rt_window_s
    mztol = args.mz_precursor_tol

    n_frags = np.zeros(C, dtype=np.int64)
    claim_sum = np.zeros(C, dtype=np.float64)
    claim_max = np.zeros(C, dtype=np.int64)
    contested_cnt = np.zeros(C, dtype=np.int64)
    contested_int = np.zeros(C, dtype=np.float64)
    total_int = np.zeros(C, dtype=np.float64)
    group_sets = [set() for _ in range(C)]

    log("[conflict] scanning claimant index")
    for p in range(len(sbin)):
        b = sbin[p]
        rt_i = srt[p]
        mz_i = smz[p]
        pmz_i = spmz[p]
        c_i = scid[p]
        tol_abs = mz_i * ppm
        rt_lo = rt_i - rtw
        rt_hi = rt_i + rtw
        others = None
        for nb in (b - 1, b, b + 1):
            se = bin_index.get(nb)
            if se is None:
                continue
            s, e = se
            seg_rt = srt[s:e]
            lo = s + int(np.searchsorted(seg_rt, rt_lo, side="left"))
            hi = s + int(np.searchsorted(seg_rt, rt_hi, side="right"))
            if hi <= lo:
                continue
            mzs = smz[lo:hi]
            pmzs = spmz[lo:hi]
            cids = scid[lo:hi]
            m = (np.abs(mzs - mz_i) <= tol_abs) & \
                (np.abs(pmzs - pmz_i) <= mztol) & \
                (cids != c_i)
            if m.any():
                if others is None:
                    others = set()
                others.update(int(x) for x in cids[m])
        cc = 0 if others is None else len(others)
        n_frags[c_i] += 1
        claim_sum[c_i] += cc
        if cc > claim_max[c_i]:
            claim_max[c_i] = cc
        total_int[c_i] += spint[p]
        if cc > 0:
            contested_cnt[c_i] += 1
            contested_int[c_i] += spint[p]
            group_sets[c_i].update(others)

    log("[conflict] reducing per-candidate features")
    with np.errstate(invalid="ignore", divide="ignore"):
        nf = n_frags.astype(np.float64)
        safe_nf = np.where(nf > 0, nf, 1.0)
        claimant_count_mean = claim_sum / safe_nf
        contested_frac = contested_cnt / safe_nf
        unique_cnt = n_frags - contested_cnt
        unique_frac = unique_cnt / safe_nf
        safe_int = np.where(total_int > 0, total_int, 1.0)
        shared_intensity_frac = contested_int / safe_int
    # Candidates with no fragments get zeroed features (documented default).
    claimant_count_mean[nf == 0] = 0.0
    contested_frac[nf == 0] = 0.0
    unique_frac[nf == 0] = 0.0
    shared_intensity_frac[total_int == 0] = 0.0
    conflict_group_size = np.array([len(g) for g in group_sets], dtype=np.int64)

    out = {
        "candidate_id": uids.astype(np.uint32),
        "claimant_count_mean": claimant_count_mean,
        "claimant_count_max": claim_max,
        "contested_fragment_count": contested_cnt,
        "contested_fragment_frac": contested_frac,
        "unique_fragment_count": unique_cnt,
        "unique_fragment_frac": unique_frac,
        "conflict_group_size": conflict_group_size,
        "shared_intensity_frac": shared_intensity_frac,
    }

    if prelim is not None:
        log("[conflict] computing candidate ambiguity (P5.5)")
        base_pep = meta["base_pep"]
        is_decoy = meta["is_decoy"]
        margin_alt = np.full(C, np.nan, dtype=np.float64)
        margin_dec = np.full(C, np.nan, dtype=np.float64)
        n_comp = np.zeros(C, dtype=np.int64)
        comp_entropy = np.zeros(C, dtype=np.float64)
        for c in range(C):
            grp = sorted(group_sets[c])  # deterministic order
            n_comp[c] = len(grp)
            if not grp:
                continue
            self_score = prelim[c]
            alt = [prelim[o] for o in grp if base_pep[o] != base_pep[c]
                   and np.isfinite(prelim[o])]
            dec = [prelim[o] for o in grp if is_decoy[o] and np.isfinite(prelim[o])]
            if alt:
                margin_alt[c] = self_score - max(alt)
            if dec:
                margin_dec[c] = self_score - max(dec)
            scores = [self_score] + [prelim[o] for o in grp]
            comp_entropy[c] = softmax_entropy(scores)
        out["margin_to_best_alt_peptide"] = margin_alt
        out["margin_to_best_decoy"] = margin_dec
        out["n_competitors_within_group"] = n_comp
        out["competitor_score_entropy"] = comp_entropy

    table = pa.table(out)
    pq.write_table(table, args.out)

    # Summary
    mean_group = float(conflict_group_size.mean()) if C else 0.0
    mean_contested = float(contested_frac.mean()) if C else 0.0
    dt = time.time() - t0
    log("[conflict] done")
    print(f"conflict.parquet written to {args.out}")
    print(f"candidates: {C}")
    print(f"transitions scanned: {nfrag}")
    print(f"mean conflict_group_size: {mean_group:.4f}")
    print(f"mean contested_fragment_frac: {mean_contested:.4f}")
    if args.max_candidates is not None:
        print(f"NOTE: --max-candidates={args.max_candidates} limits both the "
              f"query set and the claimant universe; counts are lower bounds.")
    print(f"runtime_s: {dt:.1f}")


if __name__ == "__main__":
    main()
