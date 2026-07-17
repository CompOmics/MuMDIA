"""Empirical transition reanalysis (a first, DIA-NN-`--reanalyse`-style pass).

From a first-pass rescored result, learn how reliably each PREDICTED-RANK transition
is actually observed in-peak among confident target IDs, then reweight (or prune)
the library fragments by that rank reliability. The transform depends only on a
transition's predicted-intensity RANK within its candidate, not on its identity, so
it is applied SYMMETRICALLY to targets and decoys and cannot by itself distort the
target-decoy null. The user then re-extracts/re-rescores with the new library.

This mirrors DIA-NN's empirical library: weak/interfered low-rank transitions are
down-weighted or dropped, high-rank signature transitions kept.

Usage:
  python empirical_reanalyse.py <scored.parquet> <chrom.parquet> <lib_frag_in.parquet>
         <lib_frag_out.parquet> [--q 0.01] [--mode reweight|prune] [--prune-thr 0.25]

Caveat (leakage): rank reliability is learned globally from all confident IDs here.
The rigorous version learns it out-of-fold / from other runs (see the audit's P3);
this global version is a usable first pass, not the leakage-free final.
"""
import re
import sys

import numpy as np
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import pyarrow as pa

strip = lambda p: re.sub(r"\[[^\]]*\]", "", str(p))


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    scored_p, chrom_p, lib_in, lib_out = sys.argv[1:5]
    q_conf = float(arg("--q", "0.01"))
    mode = arg("--mode", "reweight")
    prune_thr = float(arg("--prune-thr", "0.25"))

    # Confident target candidates from the first pass.
    sc = pds.dataset(scored_p).to_table(columns=["candidate_id", "label", "q_value"]).to_pandas()
    conf = set(
        sc.candidate_id[(sc.label == "target") & (sc.q_value <= q_conf)].astype(int)
    )
    if not conf:
        raise SystemExit("empirical_reanalyse: no confident target IDs at the given q")

    # Library fragments; rank per candidate by predicted intensity (0 = strongest).
    lf = pds.dataset(lib_in).to_table().to_pandas()
    lf = lf.sort_values(["candidate_id", "predicted_intensity"], ascending=[True, False])
    lf["rank"] = lf.groupby("candidate_id").cumcount()

    # Observed-in-peak set from the chromatograms: a (candidate_id, frag_name) whose
    # XIC has any nonzero intensity is "present".
    ch = pds.dataset(chrom_p).to_table(columns=["candidate_id", "frag_name", "intensity"]).to_pandas()
    ch = ch[ch.candidate_id.astype(int).isin(conf)]
    present = set()
    for cid, fn, inten in zip(ch.candidate_id, ch.frag_name, ch.intensity):
        arr = np.asarray(inten, dtype=float)
        if arr.size and arr.max() > 0.0:
            present.add((int(cid), str(fn)))

    # Rank reliability = fraction of confident candidates whose rank-r transition is
    # present in-peak (denominator = confident candidates that HAVE a rank-r transition).
    conf_lf = lf[lf.candidate_id.astype(int).isin(conf)]
    max_rank = int(conf_lf["rank"].max())
    retention = np.ones(max_rank + 1)
    for r in range(max_rank + 1):
        rows = conf_lf[conf_lf["rank"] == r]
        if len(rows) == 0:
            continue
        obs = sum((int(c), str(n)) in present for c, n in zip(rows.candidate_id, rows.name))
        retention[r] = obs / len(rows)
    print("rank reliability (fraction observed in-peak among confident IDs):")
    for r in range(min(max_rank + 1, 15)):
        print(f"  rank {r:2d}: {retention[r]:.3f}")

    # Apply symmetrically by rank to the WHOLE library (targets + decoys).
    rk = lf["rank"].to_numpy().clip(0, max_rank)
    rel = retention[rk]
    if mode == "prune":
        keep = rel >= prune_thr
        lf = lf[keep].copy()
        print(f"prune: kept {int(keep.sum())}/{len(keep)} transitions (rank reliability >= {prune_thr})")
    else:
        lf = lf.copy()
        lf["predicted_intensity"] = (lf["predicted_intensity"].to_numpy() * rel).astype(np.float32)
        print(f"reweight: scaled {len(lf)} predicted intensities by rank reliability")

    lf = lf.drop(columns=["rank"]).sort_values("candidate_id").reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(lf, preserve_index=False), lib_out)
    print(f"wrote empirical library -> {lib_out} ({len(lf)} fragments)")


if __name__ == "__main__":
    main()
