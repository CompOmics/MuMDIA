"""One-off: recompute predicted_irt in a fragment_library_precursors.parquet
using the corrected DeepLC multitask aggregation, without rebuilding fragments.

Usage: python patch_irt.py <precursors.parquet>
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    path = sys.argv[1]
    import deeplc

    tbl = pq.read_table(path)
    pforms = tbl.column("peptidoform").to_pylist()

    # unique peptidoforms (RT is charge-independent)
    uniq = {}
    order = []
    for pf in pforms:
        if pf not in uniq:
            uniq[pf] = None
            order.append(pf)

    preds = {}
    chunk = 200_000
    for start in range(0, len(order), chunk):
        batch = order[start:start + chunk]
        a = np.asarray(deeplc.predict(batch), dtype=np.float64)
        if a.ndim == 2:
            a = a.mean(axis=1)
        for pf, v in zip(batch, a):
            preds[pf] = float(v)
        print(f"  {min(start + chunk, len(order))}/{len(order)} unique peptidoforms")

    new_irt = np.array([preds[pf] for pf in pforms], dtype=np.float32)
    idx = tbl.schema.get_field_index("predicted_irt")
    tbl = tbl.set_column(idx, "predicted_irt", pa.array(new_irt, pa.float32()))
    pq.write_table(tbl, path)
    print(f"patched predicted_irt for {len(pforms)} rows in {path}")


if __name__ == "__main__":
    main()
