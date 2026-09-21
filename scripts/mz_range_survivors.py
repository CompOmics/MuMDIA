"""Candidate ids whose precursor m/z lies inside the run's isolation range, written as a
prescan-style survivors table.

Usage: python mz_range_survivors.py <lib_precursors> <isolation_windows> <out_survivors> [tol_mz]

A precursor no isolation window can select never yields a fragment, so it is dead weight in
every stage: on the 9-mer immunopeptidomics library 41% of the targets are charge 1 at m/z
above the run's 723 upper edge. Feed the output to assemble_survivors.py, which keeps the
target/decoy pair together (both share the precursor m/z) and renumbers candidate_id.
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    lib, iw, out = sys.argv[1:4]
    tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    w = pq.read_table(iw, columns=["lower", "upper"]).to_pandas()
    lo, hi = w["lower"].to_numpy(dtype=float), w["upper"].to_numpy(dtype=float)
    order = np.argsort(lo)
    merged = []
    for a, b in zip(lo[order], hi[order]):
        if merged and a <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    mz = pq.read_table(lib, columns=["precursor_mz"]).column(0).to_numpy().astype(float)
    keep = np.zeros(len(mz), dtype=bool)
    for a, b in merged:
        keep |= (mz >= a - tol) & (mz <= b + tol)
    ids = np.flatnonzero(keep).astype(np.uint32)
    pq.write_table(pa.table({"candidate_id": pa.array(ids, pa.uint32())}), out, compression="snappy")
    print(f"{len(ids)} of {len(mz)} precursors ({len(ids) / max(1, len(mz)):.1%}) inside {len(merged)} merged "
          f"isolation intervals [{merged[0][0]:.1f}, {merged[-1][1]:.1f}] (tol {tol}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
