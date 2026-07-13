"""DeepLC sidecar worker (PLAN.md Section 3.2 file contract).

Usage:
    python deeplc_worker.py <input.parquet> <output.parquet>

Input parquet columns:  id (uint32), peptidoform (ProForma string)
Output parquet columns: id (uint32), predicted_rt (float)

Run with the env that has DeepLC 4.0 (PR #99 multitask, deeplc_v4_pt). Uses the
default multitask model, uncalibrated (the per-run LOESS calibration in
rt-im-train maps these predictions onto observed RT).
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    import deeplc

    tbl = pq.read_table(in_path)
    ids = tbl.column("id").to_pylist()
    pforms = tbl.column("peptidoform").to_pylist()

    preds = np.empty(len(pforms), dtype=np.float32)
    chunk = 200_000
    for start in range(0, len(pforms), chunk):
        end = min(start + chunk, len(pforms))
        p = np.asarray(deeplc.predict(pforms[start:end]), dtype=np.float64)
        # The multitask model returns an ensemble matrix (N, n_models); average
        # across models to get a single RT prediction per peptide.
        if p.ndim == 2:
            p = p.mean(axis=1)
        preds[start:end] = p.astype(np.float32)

    out = pa.table({
        "id": pa.array(ids, pa.uint32()),
        "predicted_rt": pa.array(preds, pa.float32()),
    })
    pq.write_table(out, out_path)
    print(f"deeplc_worker: {len(ids)} peptides predicted")


if __name__ == "__main__":
    main()
