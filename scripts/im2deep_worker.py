"""IM2Deep sidecar worker (the file contract in docs/13_sidecars.md).

Usage:
    python im2deep_worker.py <input.parquet> <output.parquet> [threads]

Input parquet columns:  id (uint32), peptidoform (ProForma string), charge (int32),
                        precursor_mz (float64)
Output parquet columns: id (uint32), ccs (float32, A^2), predicted_im (float64, 1/K0 in
                        V s cm^-2)

The single-conformer model, uncalibrated. Each (peptidoform, charge) gets one CCS, which is
converted to 1/K0 with IM2Deep's own Mason-Schamp conversion (N2) at the precursor m/z the
engine sends, so `mumdia_core::constants::im_to_ccs` inverts it to rounding. A value that is
not a finite positive 1/K0 stops the worker: the engine would refuse it anyway.

`threads` (optional, the engine passes its own thread count) sets torch's CPU pool, capped at
the physical cores available to the process, as the DeepLC workers cap theirs;
MUMDIA_IM2DEEP_THREAD_CAP=N sets the cap and 0 disables it.
"""
import os
import sys

# `im2deep` first, before numpy/pyarrow: it is torch-backed like DeepLC, and on Windows the
# other order aborts torch's DLL initialisation (see deeplc_worker.py). Load-bearing order.
import im2deep  # noqa: E402

# The engine's floor (`mumdia_core::constants::MIN_IM2DEEP_VERSION`), repeated here because
# the worker can be run by hand: 2.x is the `im2deep.predict` / `im2deep.ccs2im` API this
# worker calls.
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
        "im2deep %s is older than the required %d.%d.%d (pip install 'im2deep>=%d.%d.%d')"
        % (getattr(im2deep, "__version__", "unknown"), *_MIN_IM2DEEP, *_MIN_IM2DEEP)
    )

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402  (already loaded by im2deep; named here to size its thread pool)
from psm_utils import PSM, PSMList  # noqa: E402

from _lib_io import write_engine_table  # noqa: E402
from _torch_threads import capped_threads, thread_cap  # noqa: E402

# Rows per `im2deep.predict` call: bounds the PSMList and feature-matrix memory.
_CHUNK = 200_000


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    asked = int(sys.argv[3]) if len(sys.argv) > 3 and int(sys.argv[3]) > 0 else 0
    cap, cap_why = thread_cap("MUMDIA_IM2DEEP_THREAD_CAP", "auto")
    base = asked if asked > 0 else torch.get_num_threads()
    use = capped_threads(base, cap)
    print("im2deep_worker: torch threads=%d (asked %s; cap %s: %s)"
          % (use, asked if asked > 0 else "torch default %d" % base,
             cap if cap > 0 else "none", cap_why), flush=True)

    tbl = pq.read_table(in_path)
    ids = tbl.column("id").to_numpy()
    pforms = tbl.column("peptidoform").to_pylist()
    charges = tbl.column("charge").to_numpy().astype(np.int64)
    mz = tbl.column("precursor_mz").to_numpy().astype(np.float64)

    # The prediction loop draws a progress bar; keep it out of a non-terminal log.
    os.environ.setdefault("TERM", "dumb")
    # `num_threads` is passed on to `torch.set_num_threads` inside IM2Deep.
    kwargs = {"batch_size": 4096, "num_threads": use}
    ccs = np.empty(len(pforms), dtype=np.float64)
    for start in range(0, len(pforms), _CHUNK):
        end = min(start + _CHUNK, len(pforms))
        psms = PSMList(psm_list=[
            PSM(peptidoform=f"{pforms[i]}/{charges[i]}", spectrum_id=str(i))
            for i in range(start, end)
        ])
        ccs[start:end] = np.asarray(im2deep.predict(psms, predict_kwargs=kwargs),
                                    dtype=np.float64)
        print(f"im2deep_worker: {end}/{len(pforms)} precursors predicted", flush=True)

    im = np.asarray(im2deep.ccs2im(ccs, mz, charges), dtype=np.float64)
    bad = ~(np.isfinite(im) & (im > 0))
    if bad.any():
        sys.exit(f"im2deep_worker: {int(bad.sum())} predictions are not a finite positive 1/K0")
    write_engine_table(pa.table({
        "id": pa.array(ids, pa.uint32()),
        "ccs": pa.array(ccs.astype(np.float32), pa.float32()),
        "predicted_im": pa.array(im, pa.float64()),
    }), out_path)
    print(f"im2deep_worker: {len(ids)} precursors written", flush=True)


if __name__ == "__main__":
    main()
