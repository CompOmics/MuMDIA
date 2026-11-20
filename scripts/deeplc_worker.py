"""DeepLC sidecar worker (the file contract in docs/13_sidecars.md).

Usage:
    python deeplc_worker.py <input.parquet> <output.parquet> [threads]

`threads` (optional, the engine passes its own thread count) sets torch's CPU pool. It is
capped at the physical cores available to the process, like the fine-tune worker's pools;
MUMDIA_DEEPLC_THREAD_CAP=N sets the cap and 0 disables it. Without `threads` torch keeps
its own default, under the same cap.

Input parquet columns:  id (uint32), peptidoform (ProForma string)
Output parquet columns: id (uint32), predicted_rt (float)

Run with an env that has DeepLC 4.5.0 or newer (the engine's floor). Uses the
default multitask model, uncalibrated (the per-run LOESS calibration in
rt-im-train maps these predictions onto observed RT).
"""
import sys
import time
# `deeplc` MUST be imported before numpy/pyarrow. DeepLC 4.x is torch-backed, and on Windows
# importing numpy (and the pyarrow that follows it) first makes torch's DLL initialisation fail
# outright:
#
#   OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
#   Error loading "...\torch\lib\c10.dll" or one of its dependencies.
#
# `deeplc_finetune.py` already ordered its imports this way for the same reason. This worker
# instead deferred `import deeplc` into main(), which put it after the module-level numpy/pyarrow
# and reproduced the crash. The bug stayed hidden because imported-library mode skips predict-frag
# entirely, so the native RT-prediction path is not exercised by the usual runs; it surfaced only
# when building a library from FASTA. The ordering is load-bearing, not stylistic -- do not sort it.
import deeplc  # noqa: F401  (imported for its side effect of loading torch first)
from _deeplc_common import (  # noqa: E402  (after deeplc, as every import here)
    MIN_DEEPLC,
    load_base_model,
    quiet_deeplc_progress,
    require_version,
)

require_version(deeplc, MIN_DEEPLC)
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch  # already loaded by deeplc; named here to size its thread pool
from _torch_threads import capped_threads, thread_cap  # noqa: E402  (after deeplc, as every import here)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    # The engine passes its thread count as a third argument; an older engine passes none
    # and torch keeps its own default. Either way the cap applies.
    asked = int(sys.argv[3]) if len(sys.argv) > 3 and int(sys.argv[3]) > 0 else 0
    cap, cap_why = thread_cap("MUMDIA_DEEPLC_THREAD_CAP", "auto")
    base = asked if asked > 0 else torch.get_num_threads()
    use = capped_threads(base, cap)
    if use != torch.get_num_threads():
        torch.set_num_threads(use)
    print("deeplc_worker: torch threads=%d (asked %s; cap %s: %s)"
          % (torch.get_num_threads(), asked if asked > 0 else "torch default %d" % base,
             cap if cap > 0 else "none", cap_why), flush=True)

    tbl = pq.read_table(in_path)
    ids = tbl.column("id").to_pylist()
    pforms = tbl.column("peptidoform").to_pylist()

    # One model for every chunk: `deeplc.predict` without `model=` reads the checkpoint
    # from disk on each call (`load_base_model`).
    t0 = time.perf_counter()
    model = load_base_model()
    t_load = time.perf_counter() - t0

    preds = np.empty(len(pforms), dtype=np.float32)
    chunk = 200_000
    with quiet_deeplc_progress():
        for start in range(0, len(pforms), chunk):
            end = min(start + chunk, len(pforms))
            p = np.asarray(deeplc.predict(pforms[start:end], model=model), dtype=np.float64)
            # The multitask model returns an ensemble matrix (N, n_models); average
            # across models to get a single RT prediction per peptide.
            if p.ndim == 2:
                p = p.mean(axis=1)
            preds[start:end] = p.astype(np.float32)

    t_predict = time.perf_counter() - t0 - t_load
    out = pa.table({
        "id": pa.array(ids, pa.uint32()),
        "predicted_rt": pa.array(preds, pa.float32()),
    })
    pq.write_table(out, out_path)
    print(f"deeplc_worker: {len(ids)} peptides predicted (model load {t_load:.1f}s, "
          f"prediction {t_predict:.1f}s)")


if __name__ == "__main__":
    main()
