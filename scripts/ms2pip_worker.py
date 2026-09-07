"""MS2PIP sidecar worker (the file contract in docs/13_sidecars.md).

Usage:
    python ms2pip_worker.py <input.parquet> <output.parquet> [model] [processes]

Input parquet columns:  id (uint32), peptidoform (ProForma string), charge (int)
Output parquet columns: id (uint32), ion_type (str 'b'/'y'), ordinal (int),
                        intensity (float, linear)

Run with an env that has ms2pip + pyarrow. MS2PIP predicts singly-charged b/y
intensities in log2 space; converted to linear here (2**x - 0.001, clipped at 0).

`processes` is the size of the MS2PIP worker pool. The engine passes its own thread
count; without the argument the historical cap of min(8, cpu_count) applies. On the
9.8M-peptidoform HYE library that cap left 24 of the 32 requested cores idle for the
whole prediction.

The output is assembled from numpy arrays per chunk rather than from four Python
lists appended one fragment at a time. At 9.8M peptidoforms the lists were about
300M appends and 13 GB of Python objects before the first byte was written.
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ION_CODES = ("b", "y")


def fragment_rows(results):
    """Flatten MS2PIP results into four aligned arrays: id, ion code, ordinal, intensity.

    `results` yields objects with `.psm.spectrum_id` (the row id as text) and
    `.predicted_intensity`, a dict ion -> log2-intensity array, possibly None. Rows come
    out in the order the per-fragment loop produced them: per result, ions b then y,
    ordinals ascending and 1-based; intensities are 2**x - 0.001 clipped at 0, computed
    in float64 and stored as float32. A result with no predictions contributes no rows,
    and the engine then falls back to its native intensities for that candidate.
    """
    ids, ions, ords, ints = [], [], [], []
    for r in results:
        pred = r.predicted_intensity
        if not pred:
            continue
        rid = int(r.psm.spectrum_id)
        for code, ion in enumerate(ION_CODES):
            arr = pred.get(ion)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size == 0:
                continue
            lin = np.clip(np.power(2.0, arr) - 0.001, 0.0, None)
            n = lin.size
            ids.append(np.full(n, rid, dtype=np.uint32))
            ions.append(np.full(n, code, dtype=np.int8))
            ords.append(np.arange(1, n + 1, dtype=np.int32))
            ints.append(lin.astype(np.float32))
    if not ids:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    return (
        np.concatenate(ids),
        np.concatenate(ions),
        np.concatenate(ords),
        np.concatenate(ints),
    )


def to_table(ids, ions, ords, ints):
    """The four arrays as the arrow table the engine reads: plain utf8 `ion_type`, not a
    dictionary column, because `TableFile::str` accepts utf8 only."""
    ion_type = pa.DictionaryArray.from_arrays(
        pa.array(ions, pa.int8()), pa.array(list(ION_CODES), pa.string())
    ).dictionary_decode()
    return pa.table(
        {
            "id": pa.array(ids, pa.uint32()),
            "ion_type": ion_type,
            "ordinal": pa.array(ords, pa.int32()),
            "intensity": pa.array(ints, pa.float32()),
        }
    )


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "HCD"
    cpu = os.cpu_count() or 1
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else min(8, cpu)
    procs = max(1, min(procs, cpu))

    from psm_utils import PSM, PSMList
    from ms2pip import predict_batch

    tbl = pq.read_table(in_path)
    ids = tbl.column("id").to_pylist()
    pforms = tbl.column("peptidoform").to_pylist()
    charges = tbl.column("charge").to_pylist()

    # Rows per predict_batch call. Each call starts one worker pool, so the chunk grows
    # with the pool: 20k rows per process keeps the pool start-up a small part of the
    # work at any process count, and 100k is the historical floor.
    chunk = max(100_000, 20_000 * procs)
    parts = []
    n_rows = 0
    for start in range(0, len(ids), chunk):
        end = min(start + chunk, len(ids))
        # __main__ guard below makes the Windows 'spawn' start method safe for the pool.
        # MS2PIP predictions are deterministic regardless of the process count.
        psms = PSMList(
            psm_list=[
                PSM(peptidoform=f"{pforms[i]}/{charges[i]}", spectrum_id=str(ids[i]))
                for i in range(start, end)
            ]
        )
        results = predict_batch(psms, model=model, processes=procs)
        part = fragment_rows(results)
        n_rows += part[0].size
        parts.append(part)
        print(f"ms2pip_worker: {end}/{len(ids)} peptidoforms predicted", flush=True)

    if not parts:
        parts = [fragment_rows([])]
    cols = [np.concatenate([p[k] for p in parts]) for k in range(4)]
    pq.write_table(to_table(*cols), out_path)
    print(f"ms2pip_worker: {len(ids)} peptidoforms -> {n_rows} fragment rows")


if __name__ == "__main__":
    main()
