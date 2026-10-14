"""DeepLC sidecar worker (the file contract in docs/13_sidecars.md).

Usage:
    python deeplc_worker.py <input.parquet> <output.parquet> [threads]

`threads` (optional, the engine passes its own thread count) sets torch's CPU pool. It is
capped at the physical cores available to the process, like the fine-tune worker's pools;
MUMDIA_DEEPLC_THREAD_CAP=N sets the cap and 0 disables it. Without `threads` torch keeps
its own default, under the same cap.

Input parquet columns:  id (uint32), peptidoform (ProForma string)
Output parquet columns: id (uint32), predicted_rt (float)

Run with the env that has DeepLC 4.0 (PR #99 multitask, deeplc_v4_pt). Uses the
default multitask model, uncalibrated (the per-run LOESS calibration in
rt-im-train maps these predictions onto observed RT).
"""
import contextlib
import io
import os
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


# The engine's floor (`mumdia_core::constants::MIN_DEEPLC_VERSION`), repeated here because
# `mumdia doctor` cannot see a version that changes under its feet. The base-model
# calibration is only sound from 4.1.1 on (4.0.0a2 memorised anchors: in-sample 15.9 s
# against held-out 195 s residuals), the multi-head calibration needs 4.4.0, and the
# projection cache, on by default, the factored prediction matrix of 4.5.0.
_MIN_DEEPLC = (4, 5, 0)


def _check_deeplc_version():
    raw = getattr(deeplc, "__version__", None)
    if raw is None:
        try:
            import importlib.metadata as _m
            raw = _m.version("deeplc")
        except Exception:  # pragma: no cover
            raw = ""
    parts = []
    for piece in str(raw).split(".")[:3]:
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    if tuple(parts) < _MIN_DEEPLC:
        sys.exit(
            "deeplc %s is older than the required %d.%d.%d (pip install 'deeplc>=%d.%d.%d')"
            % (raw, *_MIN_DEEPLC, *_MIN_DEEPLC)
        )


_check_deeplc_version()
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch  # already loaded by deeplc; named here to size its thread pool
from _torch_threads import capped_threads, thread_cap  # noqa: E402  (after deeplc, as every import here)


def load_base_model():
    """The DeepLC base model, loaded once, or None to let every call load its own.

    `deeplc.predict(batch)` loads the checkpoint from disk on every call, and
    `predict_and_calibrate` twice (once to size its head source, once to predict), so a
    whole-library prediction in 100,000-peptidoform chunks read the model 50 to 100 times.
    `load_model` hands a module instance back unchanged, and prediction runs in eval mode
    under `no_grad`, so passing the one instance gives the same numbers (measured
    bit-identical on 3,002 peptidoforms, DeepLC 4.5.0). The helpers are private DeepLC API
    (present in 4.4.0 and 4.5.0) and the engine sets no DeepLC ceiling, so a release that
    moves, renames or re-signatures them returns None here, with a warning, and every call
    loads its own model as before: slower, same numbers. Identical in both DeepLC workers.
    """
    try:
        from deeplc import _model_ops
        from deeplc.core import DEFAULT_MODEL

        return _model_ops.load_model(DEFAULT_MODEL)
    except (ImportError, AttributeError, TypeError) as exc:
        print("WARNING: could not load the DeepLC base model once (%s: %s); every "
              "prediction call loads its own" % (type(exc).__name__, exc), flush=True)
        return None


class _DropBlankProgress(io.TextIOBase):
    """`sys.stdout` with DeepLC's empty progress writes removed.

    DeepLC's prediction loop writes a carriage-return progress indicator that renders as
    nothing when stdout is not a terminal, so every update arrives as a bare `\r\n`.
    Measured on a real library build: 5,697 blank lines from a 2.9M-peptide prediction,
    99% of the entire log. The engine INHERITS a worker's stdout rather than capturing it
    (`sidecar.rs::run_worker`, deliberately, so long-running progress reaches the user
    live), so those lines land in the terminal, in any redirected log file, and in the
    desktop application's run log, where they push the real messages out of view.

    Only segments that are empty once carriage returns and whitespace are stripped get
    dropped. Anything DeepLC actually says still comes through, in order, and stderr --
    where a traceback goes -- is not touched at all. Set `MUMDIA_DEEPLC_RAW_OUTPUT=1` to
    disable the filter when debugging the worker itself.

    `\r` counts as a terminator as well as `\n`: a progress writer that never emits a
    newline would otherwise accumulate in the buffer for the whole run.
    """

    def __init__(self, inner):
        self._inner = inner
        self._buf = ""

    def write(self, s):
        self._buf += s
        while True:
            i = min(
                (p for p in (self._buf.find("\n"), self._buf.find("\r")) if p >= 0),
                default=-1,
            )
            if i < 0:
                break
            line, self._buf = self._buf[:i], self._buf[i + 1 :]
            if line.strip():
                self._inner.write(line + "\n")
                self._inner.flush()
        return len(s)

    def flush(self):
        self._inner.flush()

    def close(self):
        # Whatever is left had no terminator; emit it if it says anything.
        if self._buf.strip():
            self._inner.write(self._buf)
        self._buf = ""
        self._inner.flush()


@contextlib.contextmanager
def quiet_deeplc_progress():
    """Install the filter for the duration of a block, unless disabled by env."""
    if os.environ.get("MUMDIA_DEEPLC_RAW_OUTPUT", "").strip() not in ("", "0"):
        yield
        return
    original = sys.stdout
    proxy = _DropBlankProgress(original)
    sys.stdout = proxy
    try:
        yield
    finally:
        proxy.close()
        sys.stdout = original


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
