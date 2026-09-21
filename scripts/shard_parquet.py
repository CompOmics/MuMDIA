"""Split a parquet table into N row-contiguous shards (row-group aligned) or concatenate shards back.

  python shard_parquet.py split <in.parquet> <out_prefix> <n_shards>
  python shard_parquet.py concat <out.parquet> <shard1.parquet> [<shard2.parquet> ...]

Row order is preserved, so a precursor table split this way and calibrated shard by shard
(deeplc_finetune.py --multihead fits the same ridge from the same seeds in every shard)
concatenates back to a table with the same candidate_id row alignment.
"""
import sys

import pyarrow.parquet as pq


def split(inp, prefix, n):
    pf = pq.ParquetFile(inp)
    rgs = pf.num_row_groups
    per = max(1, -(-rgs // n))
    k = 0
    for start in range(0, rgs, per):
        out = f"{prefix}{k:02d}.parquet"
        w = None
        rows = 0
        for rg in range(start, min(start + per, rgs)):
            t = pf.read_row_group(rg)
            if w is None:
                w = pq.ParquetWriter(out, t.schema, compression="snappy")
            w.write_table(t)
            rows += t.num_rows
        w.close()
        print(f"shard {k}: row groups {start}-{min(start + per, rgs) - 1}, {rows} rows -> {out}", flush=True)
        k += 1


def concat(out, shards):
    w = None
    rows = 0
    for s in shards:
        pf = pq.ParquetFile(s)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            if w is None:
                w = pq.ParquetWriter(out, t.schema, compression="snappy")
            w.write_table(t)
            rows += t.num_rows
    w.close()
    print(f"concatenated {len(shards)} shards, {rows} rows -> {out}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "concat":
        concat(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
