"""Modification-localization competition features (first pass).

Implements sensitivity_plan backlog item P6.2 as a non-invasive pass over
existing MuMDIA artifacts. Peptidoforms that share a stripped sequence, a
modification multiset, and a charge but place the modifications on different
sites are localization variants. When such variants co-elute they are a
localization-ambiguity group, and site-determining ions (fragments whose m/z is
unique to one variant) distinguish them.

The engine is not modified. This is a first-pass ("fix later") implementation:
correctness, bounded memory, and a working smoke test are prioritised over
completeness. Computation is deterministic (stable sorts, sorted iteration; no
RNG). The large chromatogram list columns are read only for the small subset of
candidates that fall in an ambiguity group, which keeps memory bounded.

Per candidate it writes to `localization.parquet`:
  is_localization_ambiguous, n_localization_variants,
  site_determining_ion_count, site_determining_ion_intensity,
  localization_confidence.

House style: literal prose, no em-dashes, no hardcoded secrets.
"""

import argparse
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MOD_RE = re.compile(r"\[[^\]]*\]")
SITE_TOL_PPM = 20.0  # tolerance for calling a fragment m/z "shared" with a sibling


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def parse_peptidoform(pep):
    """Parse a ProForma-lite peptidoform into (is_decoy, stripped_sequence,
    modification_multiset, variant_signature).

    The multiset is the sorted tuple of modification tokens ignoring position.
    The signature is the sorted tuple of (residue_index, token) pairs, which
    distinguishes localization variants that share a multiset."""
    is_decoy = pep.startswith("DECOY_")
    core = pep[6:] if is_decoy else pep
    stripped = []
    mods = []          # (residue_index, token)
    i = 0
    res_idx = -1       # index of the most recent residue; -1 == N-terminal
    n = len(core)
    while i < n:
        ch = core[i]
        if ch == "[":
            j = core.find("]", i)
            if j == -1:
                # Malformed; treat the rest as sequence and stop mod parsing.
                stripped.append(core[i:])
                break
            token = core[i + 1:j]
            mods.append((res_idx, token))
            i = j + 1
        else:
            stripped.append(ch)
            res_idx += 1
            i += 1
    stripped_seq = "".join(stripped)
    multiset = tuple(sorted(t for _, t in mods))
    signature = tuple(sorted(mods))
    return is_decoy, stripped_seq, multiset, signature


def load_fragment_subset(path, wanted_ids, batch_size=1_000_000):
    """Stream chrom and collect (frag_mz, observed_apex_intensity) per candidate
    for candidates in `wanted_ids`. Observed apex intensity is the maximum of the
    per-fragment intensity trace. Bounded memory: only the wanted subset is held."""
    store = defaultdict(list)  # candidate_id -> list[(frag_mz, obs_intensity)]
    if len(wanted_ids) == 0:
        return store
    wanted = np.asarray(sorted(wanted_ids), dtype=np.uint32)
    pf = pq.ParquetFile(path)
    cols = ["candidate_id", "frag_mz", "intensity"]
    for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
        cid = batch.column("candidate_id").to_numpy(zero_copy_only=False)
        keep = np.isin(cid, wanted)
        if not keep.any():
            continue
        idx = np.nonzero(keep)[0]
        frag_mz_col = batch.column("frag_mz")
        int_col = batch.column("intensity")
        for k in idx.tolist():
            c = int(cid[k])
            mz = float(frag_mz_col[k].as_py())
            ints = int_col[k].as_py()
            obs = float(max(ints)) if ints else 0.0
            store[c].append((mz, obs))
    return store


def unique_ion_evidence(self_mz, self_int, sibling_mz):
    """Count and sum-intensity of self fragments whose m/z does not match any
    sibling fragment m/z within SITE_TOL_PPM (site-determining ions)."""
    if len(self_mz) == 0:
        return 0, 0.0
    if len(sibling_mz) == 0:
        # All fragments are unique when there are no sibling fragments.
        return len(self_mz), float(np.sum(self_int))
    sib = np.sort(np.asarray(sibling_mz, dtype=np.float64))
    count = 0
    inten = 0.0
    for mz, it in zip(self_mz, self_int):
        tol = mz * SITE_TOL_PPM * 1e-6
        pos = np.searchsorted(sib, mz)
        matched = False
        if pos < len(sib) and abs(sib[pos] - mz) <= tol:
            matched = True
        elif pos > 0 and abs(sib[pos - 1] - mz) <= tol:
            matched = True
        if not matched:
            count += 1
            inten += it
    return count, float(inten)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psms", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--out", default="localization.parquet")
    ap.add_argument("--rt-window-s", type=float, default=30.0)
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="cap the candidate set (for fast smoke tests)")
    args = ap.parse_args()

    t0 = time.time()
    log(f"[loc] loading psms {args.psms}")
    t = pq.read_table(args.psms,
                      columns=["candidate_id", "apex_rt", "charge", "peptidoform"])
    cid = t.column("candidate_id").to_numpy()
    order = np.argsort(cid, kind="stable")
    cid = cid[order]
    apex_rt = t.column("apex_rt").to_numpy()[order]
    charge = t.column("charge").to_numpy()[order]
    pep = np.asarray(t.column("peptidoform").to_pylist(), dtype=object)[order]

    if args.max_candidates is not None and args.max_candidates < len(cid):
        cid = cid[:args.max_candidates]
        apex_rt = apex_rt[:args.max_candidates]
        charge = charge[:args.max_candidates]
        pep = pep[:args.max_candidates]

    C = len(cid)
    log(f"[loc] {C} candidates; grouping by (decoy, stripped_seq, multiset, charge)")

    # Group candidates by (is_decoy, stripped_seq, multiset, charge).
    groups = defaultdict(list)  # key -> list of local indices
    signatures = [None] * C
    for i in range(C):
        is_decoy, stripped, multiset, sig = parse_peptidoform(str(pep[i]))
        signatures[i] = sig
        key = (is_decoy, stripped, multiset, int(charge[i]))
        groups[key].append(i)

    # Per-candidate outputs (defaults for non-ambiguous candidates).
    is_amb = np.zeros(C, dtype=bool)
    n_variants = np.ones(C, dtype=np.int64)
    # cluster[i] = list of local indices co-eluting with i (including i)
    clusters = [None] * C

    rtw = args.rt_window_s
    n_amb_groups = 0
    for key, members in groups.items():
        if len(members) < 2:
            continue
        # Distinct signatures present in the whole key-group.
        if len({signatures[m] for m in members}) < 2:
            continue  # only one localization variant; not ambiguous
        m_arr = np.array(members)
        rts = apex_rt[m_arr]
        sord = np.argsort(rts, kind="stable")
        m_sorted = m_arr[sord]
        rts_sorted = rts[sord]
        group_has_amb = False
        for a in range(len(m_sorted)):
            i = int(m_sorted[a])
            rt_i = rts_sorted[a]
            lo = np.searchsorted(rts_sorted, rt_i - rtw, side="left")
            hi = np.searchsorted(rts_sorted, rt_i + rtw, side="right")
            cluster_local = [int(m_sorted[b]) for b in range(lo, hi)]
            distinct_sigs = {signatures[j] for j in cluster_local}
            if len(distinct_sigs) > 1:
                is_amb[i] = True
                n_variants[i] = len(distinct_sigs)
                clusters[i] = cluster_local
                group_has_amb = True
        if group_has_amb:
            n_amb_groups += 1

    amb_ids = {int(cid[i]) for i in range(C) if is_amb[i]}
    log(f"[loc] {len(amb_ids)} candidates in ambiguity groups "
        f"across {n_amb_groups} groups")

    sd_count = np.zeros(C, dtype=np.int64)
    sd_inten = np.zeros(C, dtype=np.float64)
    loc_conf = np.ones(C, dtype=np.float64)  # non-ambiguous default: fully localized

    if amb_ids:
        log(f"[loc] streaming chrom for {len(amb_ids)} candidates")
        frag_store = load_fragment_subset(args.chrom, amb_ids)
        cid_to_local = {int(cid[i]): i for i in range(C) if is_amb[i]}

        # Per-candidate self fragment arrays.
        self_mz = {}
        self_int = {}
        for c, lst in frag_store.items():
            if lst:
                arr = np.array(lst, dtype=np.float64)
                self_mz[c] = arr[:, 0]
                self_int[c] = arr[:, 1]
            else:
                self_mz[c] = np.zeros(0)
                self_int[c] = np.zeros(0)

        # Site-determining evidence per ambiguous candidate.
        for c, i in cid_to_local.items():
            cluster_local = clusters[i] or [i]
            siblings = [j for j in cluster_local
                        if signatures[j] != signatures[i]]
            sib_mz_parts = []
            for j in siblings:
                jc = int(cid[j])
                if jc in self_mz and len(self_mz[jc]):
                    sib_mz_parts.append(self_mz[jc])
            sib_mz = np.concatenate(sib_mz_parts) if sib_mz_parts else np.zeros(0)
            smz = self_mz.get(c, np.zeros(0))
            sint = self_int.get(c, np.zeros(0))
            cnt, inten = unique_ion_evidence(smz, sint, sib_mz)
            sd_count[i] = cnt
            sd_inten[i] = inten

        # Localization confidence = self site-determining intensity over the
        # cluster sum. First pass: reuse each member's own site-determining
        # intensity (computed against its own siblings).
        for c, i in cid_to_local.items():
            cluster_local = clusters[i] or [i]
            denom = 0.0
            for j in cluster_local:
                denom += sd_inten[j]
            if denom > 0:
                loc_conf[i] = sd_inten[i] / denom
            else:
                loc_conf[i] = float("nan")

    out = pa.table({
        "candidate_id": cid.astype(np.uint32),
        "is_localization_ambiguous": is_amb,
        "n_localization_variants": n_variants,
        "site_determining_ion_count": sd_count,
        "site_determining_ion_intensity": sd_inten,
        "localization_confidence": loc_conf,
    })
    pq.write_table(out, args.out)

    n_amb = int(is_amb.sum())
    mean_variants = float(n_variants[is_amb].mean()) if n_amb else 0.0
    dt = time.time() - t0
    log("[loc] done")
    print(f"localization.parquet written to {args.out}")
    print(f"candidates: {C}")
    print(f"ambiguity groups: {n_amb_groups}")
    print(f"ambiguous candidates: {n_amb}")
    print(f"mean variants per ambiguous candidate: {mean_variants:.4f}")
    if args.max_candidates is not None:
        print(f"NOTE: --max-candidates={args.max_candidates} limits the "
              f"candidate set; ambiguity detection is over the subset only.")
    print(f"runtime_s: {dt:.1f}")


if __name__ == "__main__":
    main()
