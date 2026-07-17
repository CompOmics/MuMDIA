#!/usr/bin/env python
"""Top-K chromatographic-peak oracle for MuMDIA (spec 02 Section 5, backlog P0.4).

The central sensitivity hypothesis is that MuMDIA commits to one chromatographic
apex too early, discarding the correct peak before the scorer sees it. This
diagnostic quantifies that opportunity non-invasively: it rebuilds each
candidate's consensus elution profile from the extracted fragment chromatograms
(chrom.parquet), enumerates its chromatographic peaks with the same semantics as
the Rust `enumerate_peaks` (rust/mumdia/crates/mumdia/src/peaks.rs), and asks

  SELF : where in the area-ranked peak list does the apex MuMDIA actually chose
         (psms.apex_rt) fall? If it is rank 1 almost always, top-K rescue offers
         little; if the true apex is often rank 2+ or in no peak, the top-K peak
         model has headroom.

  REFERENCE (optional, --diann): for precursors also identified by DIA-NN, is the
         DIA-NN apex RT within tolerance of one of the top-K MuMDIA peaks? This is
         a peak-recall upper bound against an external reference.

The script reads existing Parquet artifacts only; it changes no engine state and
is deterministic (fixed numeric order, no RNG).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

# Deterministic: no stochastic component, but seed numpy for defensiveness.
np.random.seed(0)

MOD_BRACKET = re.compile(r"\[[^\]]*\]")   # ProForma-lite [UniMod:xx] / [+15.99]
MOD_PAREN = re.compile(r"\([^)]*\)")      # DIA-NN (UniMod:xx)


# --------------------------------------------------------------------------- #
# Peak enumeration: faithful port of rust/.../peaks.rs::enumerate_peaks
# --------------------------------------------------------------------------- #
def enumerate_peaks(profile, k, bound_fraction, min_prominence_frac):
    """Return peak groups as dicts, area-ranked (rank 0 = strongest).

    Each group: apex_idx, start_idx, end_idx, apex_intensity, area, rank.
    Semantics match the Rust reference exactly (local maxima with strict left /
    non-strict right, prominence floor, fractional-height boundary walk that also
    stops at a valley, dedup of maxima inside a stronger envelope, area sort with
    apex-intensity then earliest-apex tie breaks).
    """
    n = len(profile)
    if k == 0 or n == 0:
        return []
    global_max = float(profile.max()) if n else 0.0
    if global_max <= 0.0:
        return []
    prom_floor = max(min_prominence_frac, 0.0) * global_max

    maxima = []
    for i in range(n):
        v = profile[i]
        if v <= 0.0 or v < prom_floor:
            continue
        left_ok = i == 0 or v > profile[i - 1]
        right_ok = i + 1 == n or v >= profile[i + 1]
        if left_ok and right_ok:
            maxima.append(i)
    if not maxima:
        return []

    peaks = []
    bf = max(bound_fraction, 0.0)
    for apex in maxima:
        apex_v = profile[apex]
        thr = bf * apex_v
        start = apex
        while start > 0:
            prev = profile[start - 1]
            if prev < thr or prev > profile[start]:
                break
            start -= 1
        end = apex
        while end + 1 < n:
            nxt = profile[end + 1]
            if nxt < thr or nxt > profile[end]:
                break
            end += 1
        area = float(profile[start:end + 1].sum())
        peaks.append(
            {
                "apex_idx": apex,
                "start_idx": start,
                "end_idx": end,
                "apex_intensity": float(apex_v),
                "area": area,
                "rank": 0,
            }
        )

    # Strongest first by area, then apex intensity, then earliest apex.
    peaks.sort(key=lambda p: (-p["area"], -p["apex_intensity"], p["apex_idx"]))
    kept = []
    for p in peaks:
        overlaps = any(
            p["apex_idx"] >= q["start_idx"] and p["apex_idx"] <= q["end_idx"]
            for q in kept
        )
        if not overlaps:
            kept.append(p)
        if len(kept) == k:
            break
    for r, p in enumerate(kept):
        p["rank"] = r
    return kept


# --------------------------------------------------------------------------- #
# Consensus elution profile
# --------------------------------------------------------------------------- #
def build_consensus(rows, top_frags, rt_round):
    """Sum the top-`top_frags` fragment traces (by predicted_intensity) onto the
    union RT axis. `rows` is a list of (predicted_intensity, rt_list, int_list).

    Returns (rt_axis[np.float64], profile[np.float32]); empty arrays if no data.
    """
    if not rows:
        return np.empty(0), np.empty(0, dtype=np.float32)
    # Select strongest predicted fragments; empty traces simply contribute nothing.
    rows_sorted = sorted(rows, key=lambda r: -r[0])[:top_frags]
    acc = defaultdict(float)
    for _predi, rt_list, int_list in rows_sorted:
        if not rt_list:
            continue
        for t, inten in zip(rt_list, int_list):
            if inten is None:
                continue
            acc[round(float(t), rt_round)] += float(inten)
    if not acc:
        return np.empty(0), np.empty(0, dtype=np.float32)
    keys = np.array(sorted(acc.keys()), dtype=np.float64)
    prof = np.array([acc[k] for k in keys], dtype=np.float32)
    return keys, prof


def locate_apex_peak(apex_rt, rt_axis, peaks, rt_tol):
    """Return the rank of the peak the given apex_rt belongs to, or None.

    First, peaks whose [start_rt, end_rt] envelope contains apex_rt (nearest by
    apex distance wins); otherwise the nearest peak apex within rt_tol.
    """
    if not peaks:
        return None
    containing = []
    for p in peaks:
        s_rt = rt_axis[p["start_idx"]]
        e_rt = rt_axis[p["end_idx"]]
        if s_rt <= apex_rt <= e_rt:
            containing.append(p)
    if containing:
        best = min(containing, key=lambda p: abs(rt_axis[p["apex_idx"]] - apex_rt))
        return best["rank"]
    best = min(peaks, key=lambda p: abs(rt_axis[p["apex_idx"]] - apex_rt))
    if abs(rt_axis[best["apex_idx"]] - apex_rt) <= rt_tol:
        return best["rank"]
    return None


# --------------------------------------------------------------------------- #
# DIA-NN reference loading
# --------------------------------------------------------------------------- #
def strip_mods(seq):
    if seq is None:
        return ""
    s = MOD_BRACKET.sub("", str(seq))
    s = MOD_PAREN.sub("", s)
    return s.strip().upper()


def load_diann(path):
    """Load a DIA-NN report (.tsv/.parquet); return dict (stripped_seq, charge)->rt_seconds.

    Column names are matched defensively; RT minutes are detected and converted by
    the caller against the MuMDIA RT range.
    """
    import pandas as pd

    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    seq_col = pick("Stripped.Sequence", "Modified.Sequence", "Precursor.Id", "Sequence")
    chg_col = pick("Precursor.Charge", "Charge")
    rt_col = pick("RT", "RT.Start", "Retention.Time", "iRT")
    if seq_col is None or chg_col is None or rt_col is None:
        raise ValueError(
            f"DIA-NN report missing required columns; found {list(df.columns)[:20]}"
        )
    out = {}
    rts = []
    for seq, chg, rt in zip(df[seq_col], df[chg_col], df[rt_col]):
        try:
            c = int(chg)
        except (ValueError, TypeError):
            continue
        key = (strip_mods(seq), c)
        try:
            rtf = float(rt)
        except (ValueError, TypeError):
            continue
        out[key] = rtf
        rts.append(rtf)
    return out, (min(rts) if rts else 0.0), (max(rts) if rts else 0.0)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psms", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--diann", default=None, help="optional DIA-NN report (.tsv/.parquet)")
    ap.add_argument("--out", default=None, help="metrics JSON output path")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="limit to the first N candidates (0 = all)")
    ap.add_argument("--top-frags", type=int, default=6,
                    help="number of strongest predicted fragments summed into the consensus")
    ap.add_argument("--bound-fraction", type=float, default=1.0 / 3.0)
    ap.add_argument("--min-prominence", type=float, default=0.05)
    ap.add_argument("--rt-tol-s", type=float, default=10.0)
    ap.add_argument("--rt-round", type=int, default=3, help="RT rounding decimals for axis union")
    ap.add_argument("--batch-size", type=int, default=100000)
    args = ap.parse_args()

    # ---- psms: candidate -> apex_rt, and (stripped_seq,charge) for reference ---
    pcols = ["candidate_id", "apex_rt", "label", "peptidoform", "charge"]
    ptab = pq.read_table(args.psms, columns=pcols)
    cand_ids = ptab.column("candidate_id").to_numpy()
    apex_rt = ptab.column("apex_rt").to_numpy()
    labels = ptab.column("label").to_pylist()
    peptidoforms = ptab.column("peptidoform").to_pylist()
    charges = ptab.column("charge").to_numpy()

    order = np.argsort(cand_ids, kind="mergesort")
    cand_ids = cand_ids[order]
    apex_rt = apex_rt[order]
    labels = [labels[i] for i in order]
    peptidoforms = [peptidoforms[i] for i in order]
    charges = charges[order]

    n_total = len(cand_ids)
    n_sel = n_total if args.max_candidates <= 0 else min(args.max_candidates, n_total)
    sel_slice = slice(0, n_sel)
    sel_ids = cand_ids[sel_slice]
    sel_set = set(int(x) for x in sel_ids)
    max_sel = int(sel_ids[-1]) if n_sel else -1
    apex_by_cand = {int(c): float(r) for c, r in zip(sel_ids, apex_rt[sel_slice])}
    seqkey_by_cand = {
        int(c): (strip_mods(pf), int(ch))
        for c, pf, ch in zip(sel_ids, peptidoforms[:n_sel], charges[sel_slice])
    }
    mumdia_rt_min = float(np.nanmin(apex_rt[sel_slice])) if n_sel else 0.0
    mumdia_rt_max = float(np.nanmax(apex_rt[sel_slice])) if n_sel else 0.0

    # ---- DIA-NN reference (optional) ------------------------------------------
    diann = None
    if args.diann:
        diann_map, drt_min, drt_max = load_diann(args.diann)
        # Minutes detection: if the DIA-NN RT span is far below the MuMDIA span
        # (seconds), assume minutes and scale by 60.
        scale = 1.0
        if drt_max > 0 and mumdia_rt_max > 0 and drt_max <= mumdia_rt_max / 5.0:
            scale = 60.0
        diann = {k: v * scale for k, v in diann_map.items()}
        print(f"[diann] loaded {len(diann)} precursors, rt span "
              f"{drt_min:.1f}-{drt_max:.1f} (scale x{scale:g} -> seconds)")

    # ---- stream chrom, grouped by candidate_id (sorted) -----------------------
    pf = pq.ParquetFile(args.chrom)
    ccols = ["candidate_id", "predicted_intensity", "rt", "intensity"]

    # accumulators
    ranks = []                     # peak rank the self apex fell into (int)
    n_no_peak = 0                  # self apex matched no enumerated peak
    n_no_chrom = 0                 # selected candidate absent from chrom / empty profile
    peaks_per_cand = []            # peak count per processed candidate
    processed = set()
    ref_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    ref_matched = 0                # candidates matched to a DIA-NN precursor with a profile

    def process_candidate(cid, rows):
        nonlocal n_no_peak, ref_matched
        rt_axis, prof = build_consensus(rows, args.top_frags, args.rt_round)
        peaks = enumerate_peaks(prof, len(prof) if len(prof) else 0,
                                args.bound_fraction, args.min_prominence)
        peaks_per_cand.append(len(peaks))
        processed.add(cid)
        # SELF
        a_rt = apex_by_cand.get(cid)
        if a_rt is not None:
            rank = locate_apex_peak(a_rt, rt_axis, peaks, args.rt_tol_s)
            if rank is None:
                n_no_peak += 1
            else:
                ranks.append(rank)
        # REFERENCE
        if diann is not None and peaks:
            key = seqkey_by_cand.get(cid)
            d_rt = diann.get(key) if key else None
            if d_rt is not None:
                ref_matched += 1
                apex_rts = [rt_axis[p["apex_idx"]] for p in peaks]  # rank-ordered
                for K in (1, 3, 5, 10):
                    topk = apex_rts[:K]
                    if any(abs(d_rt - ar) <= args.rt_tol_s for ar in topk):
                        ref_hits[K] += 1

    cur_cid = None
    cur_rows = []
    stop = False
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=ccols):
        cids = batch.column("candidate_id").to_numpy()
        if len(cids) == 0:
            continue
        if int(cids[0]) > max_sel:
            break  # sorted; nothing selected remains
        predi = batch.column("predicted_intensity").to_numpy()
        rt_lists = batch.column("rt").to_pylist()
        int_lists = batch.column("intensity").to_pylist()
        for j in range(len(cids)):
            cid = int(cids[j])
            if cid != cur_cid:
                if cur_cid is not None and cur_cid in sel_set:
                    process_candidate(cur_cid, cur_rows)
                if cur_cid is not None and cur_cid > max_sel:
                    stop = True
                    break
                cur_cid = cid
                cur_rows = []
            if cid in sel_set:
                cur_rows.append((float(predi[j]), rt_lists[j], int_lists[j]))
        if stop:
            break
    # flush last
    if not stop and cur_cid is not None and cur_cid in sel_set:
        process_candidate(cur_cid, cur_rows)

    # selected candidates never seen in chrom
    n_no_chrom = n_sel - len(processed)

    # ---- metrics --------------------------------------------------------------
    ranks_arr = np.array(ranks, dtype=int)
    # denominator: all selected candidates that had a psms apex_rt (all of them)
    denom = n_sel
    n_rank1 = int((ranks_arr == 0).sum())
    n_top3 = int((ranks_arr < 3).sum())
    n_top5 = int((ranks_arr < 5).sum())
    n_top10 = int((ranks_arr < 10).sum())
    # "no enumerated peak" = matched to no peak OR no chrom/empty profile at all
    n_no_peak_total = n_no_peak + n_no_chrom
    ppc = np.array(peaks_per_cand, dtype=float)

    metrics = {
        "inputs": {
            "psms": os.path.abspath(args.psms),
            "chrom": os.path.abspath(args.chrom),
            "diann": os.path.abspath(args.diann) if args.diann else None,
        },
        "params": {
            "max_candidates": args.max_candidates,
            "top_frags": args.top_frags,
            "bound_fraction": args.bound_fraction,
            "min_prominence": args.min_prominence,
            "rt_tol_s": args.rt_tol_s,
        },
        "n_candidates_total": int(n_total),
        "n_candidates_selected": int(n_sel),
        "n_candidates_processed": int(len(processed)),
        "n_no_chrom_or_empty": int(n_no_chrom),
        "self": {
            "denominator": int(denom),
            "n_apex_matched_to_peak": int(len(ranks_arr)),
            "frac_rank1": n_rank1 / denom if denom else 0.0,
            "frac_top3": n_top3 / denom if denom else 0.0,
            "frac_top5": n_top5 / denom if denom else 0.0,
            "frac_top10": n_top10 / denom if denom else 0.0,
            "frac_no_peak": n_no_peak_total / denom if denom else 0.0,
            "mean_peaks_per_candidate": float(ppc.mean()) if ppc.size else 0.0,
            "median_peaks_per_candidate": float(np.median(ppc)) if ppc.size else 0.0,
            "frac_ge2_peaks": float((ppc >= 2).mean()) if ppc.size else 0.0,
        },
    }
    if diann is not None:
        metrics["reference"] = {
            "n_reference_matched": int(ref_matched),
            "reference_apex_in_top_1": ref_hits[1] / ref_matched if ref_matched else 0.0,
            "reference_apex_in_top_3": ref_hits[3] / ref_matched if ref_matched else 0.0,
            "reference_apex_in_top_5": ref_hits[5] / ref_matched if ref_matched else 0.0,
            "reference_apex_in_top_10": ref_hits[10] / ref_matched if ref_matched else 0.0,
        }

    # ---- report ---------------------------------------------------------------
    s = metrics["self"]
    print("\n=== Top-K peak oracle (SELF) ===")
    print(f"selected candidates      : {n_sel} (processed {len(processed)}, "
          f"no chrom/empty {n_no_chrom})")
    print(f"mean peaks / candidate   : {s['mean_peaks_per_candidate']:.2f} "
          f"(median {s['median_peaks_per_candidate']:.0f})")
    print(f"candidates with >=2 peaks: {s['frac_ge2_peaks']*100:.1f}%  "
          f"(the top-K opportunity size)")
    print(f"chosen apex is peak rank1: {s['frac_rank1']*100:.1f}%")
    print(f"          within top-3   : {s['frac_top3']*100:.1f}%")
    print(f"          within top-5   : {s['frac_top5']*100:.1f}%")
    print(f"          within top-10  : {s['frac_top10']*100:.1f}%")
    print(f"          in NO peak     : {s['frac_no_peak']*100:.1f}%")
    if diann is not None:
        r = metrics["reference"]
        print("\n=== Reference-apex recall (DIA-NN) ===")
        print(f"matched precursors       : {r['n_reference_matched']}")
        for K in (1, 3, 5, 10):
            print(f"DIA-NN apex in top-{K:<2}   : "
                  f"{r[f'reference_apex_in_top_{K}']*100:.1f}%")

    out = args.out or (os.path.splitext(args.psms)[0] + ".topk_oracle.json")
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n[written] {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
