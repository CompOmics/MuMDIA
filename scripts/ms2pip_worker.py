"""MS2PIP sidecar worker (the file contract in docs/13_sidecars.md).

Usage:
    python ms2pip_worker.py <input.parquet> <output.parquet> [model]

Input parquet columns:  id (uint32), peptidoform (ProForma string), charge (int)
Output parquet columns: id (uint32), ion_type (str 'b'/'y'), ordinal (int),
                        intensity (float, linear)

Run with an env that has ms2pip + pyarrow (e.g. py312_mumdia). MS2PIP predicts
singly-charged b/y intensities in log2 space; converted to linear here.
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "HCD"

    from psm_utils import PSM, PSMList
    from ms2pip import predict_batch

    tbl = pq.read_table(in_path)
    ids = tbl.column("id").to_pylist()
    pforms = tbl.column("peptidoform").to_pylist()
    charges = tbl.column("charge").to_pylist()

    out_id, out_ion, out_ord, out_int = [], [], [], []

    chunk = 100_000
    for start in range(0, len(ids), chunk):
        end = min(start + chunk, len(ids))
        psms = PSMList(psm_list=[
            PSM(peptidoform=f"{pforms[i]}/{charges[i]}", spectrum_id=str(ids[i]))
            for i in range(start, end)
        ])
        # __main__ guard above makes Windows 'spawn' safe, so use several
        # processes. MS2PIP predictions are deterministic regardless of count.
        import os
        procs = min(8, os.cpu_count() or 1)
        results = predict_batch(psms, model=model, processes=procs)
        for r in results:
            rid = int(r.psm.spectrum_id)
            pred = r.predicted_intensity  # dict {'b': array(log2), 'y': array(log2)}
            for ion in ("b", "y"):
                arr = pred.get(ion)
                if arr is None:
                    continue
                lin = np.power(2.0, np.asarray(arr, dtype=np.float64)) - 0.001
                lin = np.clip(lin, 0.0, None)
                for k, v in enumerate(lin):
                    out_id.append(rid)
                    out_ion.append(ion)
                    out_ord.append(k + 1)  # ordinal is 1-based
                    out_int.append(float(v))

    out = pa.table({
        "id": pa.array(out_id, pa.uint32()),
        "ion_type": pa.array(out_ion, pa.string()),
        "ordinal": pa.array(out_ord, pa.int32()),
        "intensity": pa.array(out_int, pa.float32()),
    })
    pq.write_table(out, out_path)
    print(f"ms2pip_worker: {len(ids)} peptidoforms -> {len(out_id)} fragment rows")


if __name__ == "__main__":
    main()
