"""Keep the N most intense fragments of every precursor of a spectral library, streaming.

Usage:
    python cap_library_fragments.py <lib_fragments.parquet> <out_fragments.parquet> [--top 12]

The engine stores one fragment offset per candidate as a 32-bit integer, so it refuses a
fragment table of more than 4,294,967,295 rows ("fragment library has N rows; per-candidate
fragment offsets are u32"). A library predicted for an unspecific (HLA) digest of a large
proteome reaches that: hundreds of millions of precursors with 15 to 20 fragments each.
Most of those fragments are weak and add little evidence; the engine's own fragment
prediction keeps the top `predict_frag.top_n_fragments` (12 in the shipped sidecar
configurations), and so does this helper for an imported library.

Per candidate the fragments are ranked by `predicted_intensity` (ties keep their stored
order) and the first `--top` are kept, in their stored order. The precursor table is not
touched: candidate ids do not change, so the engine pairs the capped fragments with the same
`lib_precursors.parquet`.

The table must be ordered by `candidate_id`, which every library writer of this repository
produces; `sort_fragments.py` fixes one that is not. One row group is resident at a time,
plus the fragments of the one candidate that straddles two row groups.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from _lib_io import narrow_table

U32_MAX = 2**32 - 1


def keep_mask(cid: np.ndarray, intensity: np.ndarray, top: int) -> np.ndarray:
    """Rows to keep: the `top` most intense of each candidate, ties in stored order.

    `cid` must be non-decreasing. A NaN intensity ranks last.
    """
    n = len(cid)
    if n == 0:
        return np.zeros(0, dtype=bool)
    key = np.where(np.isnan(intensity), -np.inf, intensity)
    # Stable sort by candidate, then by descending intensity; the row index breaks ties.
    order = np.lexsort((np.arange(n), -key, cid))
    sorted_cid = cid[order]
    starts = np.flatnonzero(np.r_[True, sorted_cid[1:] != sorted_cid[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, n]))
    rank = np.arange(n) - group_start
    keep = np.zeros(n, dtype=bool)
    keep[order[rank < top]] = True
    return keep


def cap(src: str, dst: str, top: int, row_group_rows: int = 1 << 20) -> tuple[int, int, int]:
    """Write `src` capped to `top` fragments per candidate to `dst`.

    Returns (rows read, rows written, candidates).
    """
    pf = pq.ParquetFile(src)
    for col in ("candidate_id", "predicted_intensity"):
        if pf.schema_arrow.get_field_index(col) < 0:
            raise SystemExit(f"{src} has no column {col!r}; is it a MuMDIA fragment library?")
    tmp = dst + ".partial"
    writer = None
    rows_in = rows_out = candidates = 0
    carry: pa.Table | None = None
    last_cid = -1
    try:
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            rows_in += t.num_rows
            if carry is not None:
                t = pa.concat_tables([carry, t])
                carry = None
            cid = t.column("candidate_id").to_numpy().astype(np.int64)
            if len(cid) == 0:
                continue
            if cid[0] < last_cid or np.any(cid[1:] < cid[:-1]):
                raise SystemExit(
                    f"{src} is not ordered by candidate_id (row group {i}). "
                    "Run sort_fragments.py on it first."
                )
            # The last candidate may continue in the next row group: hold it back.
            if i + 1 < pf.num_row_groups:
                tail = int(np.searchsorted(cid, cid[-1], side="left"))
                carry = t.slice(tail)
                t = t.slice(0, tail)
                cid = cid[:tail]
            if t.num_rows == 0:
                continue
            last_cid = int(cid[-1])
            inten = t.column("predicted_intensity").to_numpy(zero_copy_only=False).astype(np.float64)
            mask = keep_mask(cid, inten, top)
            candidates += int(np.count_nonzero(np.r_[True, cid[1:] != cid[:-1]]))
            out = narrow_table(t.filter(pa.array(mask)))
            if writer is None:
                writer = pq.ParquetWriter(tmp, out.schema, compression="snappy")
            writer.write_table(out, row_group_size=row_group_rows)
            rows_out += out.num_rows
        if writer is None:
            # An empty input still gets a valid, empty output with its schema.
            writer = pq.ParquetWriter(tmp, narrow_table(pf.schema_arrow.empty_table()).schema, compression="snappy")
        writer.close()
        writer = None
        pf.close()
        os.replace(tmp, dst)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp):
            os.remove(tmp)
    return rows_in, rows_out, candidates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fragments", help="input lib_fragments.parquet, ordered by candidate_id")
    ap.add_argument("out", help="output fragment table (the input is never overwritten in place)")
    ap.add_argument("--top", type=int, default=12, help="fragments kept per precursor (default 12)")
    a = ap.parse_args()
    if a.top < 1:
        raise SystemExit("--top must be at least 1")
    if os.path.abspath(a.fragments) == os.path.abspath(a.out):
        raise SystemExit("write to a new file; the input is read while the output is written")
    rows_in, rows_out, candidates = cap(a.fragments, a.out, a.top)
    print(
        f"{rows_in} fragment rows -> {rows_out} ({100.0 * rows_out / max(1, rows_in):.1f}%) "
        f"for {candidates} candidates, top {a.top} per candidate -> {a.out}",
        flush=True,
    )
    if rows_out > U32_MAX:
        print(
            f"warning: {rows_out} rows is still above the engine's limit of {U32_MAX}; "
            "lower --top or shrink the precursor table",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
