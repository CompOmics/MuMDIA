"""IM2Deep sidecar worker (the file contract in docs/13_sidecars.md).

Usage:
    python im2deep_worker.py <input.parquet> <output.parquet> <threads>

Input parquet columns:  id (uint32), peptidoform (ProForma string), charge (int32),
                        precursor_mz (float64)
Output parquet columns: id (uint32), ccs (float32, A^2), predicted_im (float64, 1/K0 in
                        V s cm^-2)

The single-conformer model (IM2DeepUni), uncalibrated: rt-im-train fits a per-run,
per-charge CCS calibration against the run's own seed anchors, the way it calibrates the
DeepLC iRT. CCS is converted to 1/K0 with IM2Deep's own Mason-Schamp conversion (N2) at
the engine's precursor m/z, so the engine's `im_to_ccs` inverts it exactly.
"""
import os
import sys

# `im2deep` first, before numpy/pyarrow: it is torch-backed like DeepLC, and on Windows the
# other order aborts torch's DLL initialisation (see deeplc_worker.py). Load-bearing order.
import im2deep  # noqa: E402

_MIN_IM2DEEP = (2, 0, 0)


def _version_tuple(raw):
    parts = []
    for piece in str(raw).split(".")[:3]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


if _version_tuple(getattr(im2deep, "__version__", "0")) < _MIN_IM2DEEP:
    sys.exit(
        "im2deep %s is older than the required %d.%d.%d (pip install 'im2deep>=2.0.0')"
        % (im2deep.__version__, *_MIN_IM2DEEP)
    )

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from psm_utils import PSM, PSMList  # noqa: E402

# Rows per `im2deep.predict` call: bounds the PSMList and feature-matrix memory.
_CHUNK = 200_000


def main():
    in_path, out_path, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tbl = pq.read_table(in_path)
    ids = tbl.column("id").to_numpy()
    pforms = tbl.column("peptidoform").to_pylist()
    charges = tbl.column("charge").to_numpy().astype(np.int64)
    mz = tbl.column("precursor_mz").to_numpy().astype(np.float64)

    # The predict loop draws a progress bar; keep it out of a non-terminal log.
    os.environ.setdefault("TERM", "dumb")
    kwargs = {"batch_size": 4096, "num_threads": max(1, threads)}
    ccs = np.empty(len(pforms), dtype=np.float64)
    for start in range(0, len(pforms), _CHUNK):
        end = min(start + _CHUNK, len(pforms))
        psms = PSMList(psm_list=[
            PSM(peptidoform=f"{pforms[i]}/{charges[i]}", spectrum_id=str(i))
            for i in range(start, end)
        ])
        ccs[start:end] = np.asarray(im2deep.predict(psms, predict_kwargs=kwargs), dtype=np.float64)
        print(f"im2deep_worker: {end}/{len(pforms)} precursors predicted", flush=True)

    im = np.asarray(im2deep.ccs2im(ccs, mz, charges), dtype=np.float64)
    bad = ~(np.isfinite(im) & (im > 0))
    if bad.any():
        sys.exit(f"im2deep_worker: {int(bad.sum())} predictions are not a finite positive 1/K0")
    pq.write_table(pa.table({
        "id": pa.array(ids, pa.uint32()),
        "ccs": pa.array(ccs.astype(np.float32), pa.float32()),
        "predicted_im": pa.array(im, pa.float64()),
    }), out_path)
    print(f"im2deep_worker: {len(ids)} precursors written")


if __name__ == "__main__":
    main()
