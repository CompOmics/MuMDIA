"""Transfer-learn (fine-tune) multitask DeepLC 4.0 on this run's confident seed
PSMs, then predict RT for all target peptidoforms. Unlike predict_and_calibrate
(a post-hoc calibration curve), finetune adapts the model weights to this run's
chromatography.

CRASH FIX (deeplc_mt env): numpy links OpenBLAS built against GNU OpenMP while
torch ships Intel OpenMP (libiomp5md.dll). Two OpenMP runtimes coexist only
because KMP_DUPLICATE_LIB_OK=TRUE suppresses the abort. With torch's default
num_threads (= all cores, e.g. 24) each runtime spawns its own full thread pool
and they oversubscribe the CPU during fine-tuning's sustained backward pass,
which crashes the machine intermittently. We pin the BLAS side to 1 thread and
bound torch to a modest pool so only one pool ever spins. Array building here is
trivial, so single-threaded BLAS costs nothing.

Usage:
  python deeplc_finetune.py <lib_t_precursors_in> <seed_psms> <lib_t_precursors_out>
                            [--threads N] [--epochs E] [--batch B] [--patience P]
                            [--max-ref N] [--predict-limit N] [--skip-predict]
  python deeplc_finetune.py <lib_t_precursors_in> - <lib_t_precursors_out> --no-finetune
      (engine path for rt_im_train.library_irt = deeplc: predict with the DeepLC base
      model, no seed needed; per-run LOESS calibration then maps the predictions onto
      observed RT)

Both torch pools (--threads for training, --predict-threads for the whole-library
prediction) are capped at the physical cores available to the process: the distinct
sysfs `core_cpus_list` sets under the affinity mask on Linux, every physical core on Windows.
MUMDIA_DEEPLC_THREAD_CAP=N sets the cap explicitly and 0 disables it. The resolved
numbers are printed and recorded under "torch_threads" in <lib_out>.summary.json.

--shards K splits the whole-library prediction across K child processes of this script
(`--shard-worker <spec.json>`, internal). The calibration or fine-tuned model is fitted
once, here, and handed to every child; each child predicts a contiguous slice of the
unique sequences cut at a multiple of the prediction chunk, so it makes exactly the calls
the single process would have made, and the parent joins the slices in order. K=1 (the
default) is the single process. See `shard_plan` for how K and the threads per child
follow from the thread budget.
"""
import os

# --- thread caps MUST be set before numpy / torch import ---
_THREADS = os.environ.get("DEEPLC_FT_THREADS", "8")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"      # still required so both runtimes load
os.environ["OMP_NUM_THREADS"] = "1"              # GNU OpenMP (OpenBLAS) -> single pool
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import contextlib
import io
import json
import pickle
import re
import shutil
import subprocess
import tempfile
import time
import deeplc                                    # import before numpy (OpenMP load order)
import sys

# The engine's default retention-time workflow calibrates DeepLC's base-model predictions
# per run without a fine-tune. That is only sound from 4.1.1 on (4.0.0a2 memorised anchors:
# in-sample 15.9 s against held-out 195 s residuals), so an older DeepLC is refused here as
# well as by `mumdia doctor`, which cannot see a version that changes under its feet.
_MIN_DEEPLC = (4, 1, 1)


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
            "deeplc %s is older than the required %d.%d.%d (pip install 'deeplc>=4.4.0')"
            % (raw, *_MIN_DEEPLC)
        )


_check_deeplc_version()
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from psm_utils import PSM, PSMList

STD = set("ACDEFGHIKLMNPQRSTVWY")
MOD_RE = r"\[[^\]]*\]"
strip_mods = lambda s: re.sub(MOD_RE, "", s)
# Strip the decoy marker before prediction: a "DECOY_" peptidoform must be predicted on
# its underlying sequence, else is_std rejects it (the '_') and the decoy keeps the base
# (un-fine-tuned) iRT, landing on a different scale than the fine-tuned targets.
base_pf = lambda s: s[6:] if s.startswith("DECOY_") else s
# The same two rules over a whole Arrow column. The decoy strip is anchored at the start,
# as `base_pf` is; `replace_substring` would strip every occurrence. `*` rather than `+`
# because `is_std("")` is true.
DECOY_PREFIX_RE = r"^DECOY_"
STD_FULL_RE = r"^[ACDEFGHIKLMNPQRSTVWY]*$"
# Unique peptidoforms per prediction call. Fixed, because the call is the unit DeepLC
# length-buckets and batches within, so the same chunks give the same numbers.
PREDICT_CHUNK = 100_000
# Torch threads per prediction shard under `--shards 0` (automatic). The survey's
# recommended layout on a 64-core host (K = 8 shards of 8 threads) and the per-shard rate
# docs/32 measured with 12 shards; not swept.
SHARD_AUTO_THREADS = 8



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

class PredictTimers:
    """Where the whole-library prediction spends its time, for `<lib_out>.summary.json`.

    DeepLC does not report its own phases, so this wraps the steps it runs:

    - featurisation: PSM parsing (`deeplc.core._parse_psms`), dataset construction
      (`DeepLCDataset.from_psm_list`), the length bucketing
      (`deeplc._model_ops._length_buckets`) and batch encoding
      (`DeepLCDataset.encode_batch`). Only the outermost of nested calls is timed.
    - forward: the model's forward calls, from a forward pre-hook and a forward hook on
      the module, synchronised first when the output is on a GPU.

    The wrappers call the original functions with the same arguments and the hooks return
    nothing, so the numbers they time are unchanged. These are private DeepLC names
    (present in 4.4.0 and 4.5.0): any that is missing leaves its total as None rather than
    failing the run. `install` restores every original on exit.
    """

    def __init__(self):
        self.featurisation = 0.0
        self.forward = 0.0
        self.have_featurisation = False
        self.have_forward = False
        self._depth = 0
        self._t_forward = None

    def _timed(self, func):
        timers = self

        def wrapped(*a, **k):
            if timers._depth:
                return func(*a, **k)
            timers._depth += 1
            t0 = time.perf_counter()
            try:
                return func(*a, **k)
            finally:
                timers.featurisation += time.perf_counter() - t0
                timers._depth -= 1

        return wrapped

    def _pre_forward(self, module, args):
        self._t_forward = time.perf_counter()

    def _post_forward(self, module, args, output):
        if self._t_forward is None:
            return
        if getattr(output, "is_cuda", False):
            torch.cuda.synchronize()
        self.forward += time.perf_counter() - self._t_forward
        self._t_forward = None

    @contextlib.contextmanager
    def install(self, model):
        restore = []
        try:
            import inspect

            from deeplc import _model_ops
            from deeplc import core as deeplc_core
            from deeplc.data import DeepLCDataset

            for owner, name in ((deeplc_core, "_parse_psms"), (_model_ops, "_length_buckets")):
                original = getattr(owner, name, None)
                if callable(original):
                    setattr(owner, name, self._timed(original))
                    restore.append((owner, name, original))
            for name in ("from_psm_list", "encode_batch"):
                static = inspect.getattr_static(DeepLCDataset, name, None)
                if isinstance(static, classmethod):
                    setattr(DeepLCDataset, name, classmethod(self._timed(static.__func__)))
                elif callable(static):
                    setattr(DeepLCDataset, name, self._timed(static))
                else:
                    continue
                restore.append((DeepLCDataset, name, static))
            self.have_featurisation = bool(restore)
        except Exception:  # noqa: BLE001 - timing is diagnostic, never a failure
            pass
        hooks = []
        if model is not None and hasattr(model, "register_forward_hook"):
            hooks = [model.register_forward_pre_hook(self._pre_forward),
                     model.register_forward_hook(self._post_forward)]
            self.have_forward = True
        try:
            yield self
        finally:
            for h in hooks:
                h.remove()
            for owner, name, original in reversed(restore):
                setattr(owner, name, original)

    def totals(self):
        return (round(self.featurisation, 3) if self.have_featurisation else None,
                round(self.forward, 3) if self.have_forward else None)


def _windows_physical_cores():
    """Physical core count from `GetLogicalProcessorInformationEx(RelationProcessorCore)`.

    One record per physical core, across every processor group, whatever its efficiency
    class: a hybrid CPU's efficiency cores are real cores for a forward pass, unlike a
    second hyperthread on a core that is already busy. None when the call fails.
    """
    try:
        import ctypes
        import struct
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        relation_processor_core = 0
        size = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(
            relation_processor_core, buf, ctypes.byref(size)
        ):
            return None
        raw = buf.raw
        cores = 0
        off = 0
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship (u32), Size (u32), payload.
        while off + 8 <= size.value:
            relationship, record_size = struct.unpack_from("<II", raw, off)
            if record_size < 8:
                return None
            if relationship == relation_processor_core:
                cores += 1
            off += record_size
        return cores or None
    except Exception:  # noqa: BLE001 - detection is best effort; None means no cap
        return None


def _sysfs_physical_cores(cpus, sysfs="/sys/devices/system/cpu"):
    """The number of physical cores behind the logical CPUs `cpus`, from sysfs, or None.

    Each CPU's `topology/core_cpus_list` (Linux 5.5 and later; `thread_siblings_list`
    before) names the logical CPUs that share its physical core, so its contents identify
    that core on every architecture, and the hyperthreads of one core count once. The
    `(physical_package_id, core_id)` pair is the fallback only: on many device-tree ARM64
    systems `core_id` restarts in each cluster and the package id is the same for every
    CPU, so the pairs of two clusters collide and eight cores count as four. None when a
    CPU has no topology at all (some containers), which leaves the count to the caller.
    """
    cores = set()
    for cpu in cpus:
        topo = "%s/cpu%d/topology/" % (sysfs, cpu)
        key = None
        for name in ("core_cpus_list", "thread_siblings_list"):
            try:
                with open(topo + name, encoding="ascii") as fh:
                    key = "cpus " + fh.read().strip()
                break
            except OSError:
                continue
        if key is None:
            try:
                with open(topo + "physical_package_id", encoding="ascii") as fh:
                    package = fh.read().strip()
                with open(topo + "core_id", encoding="ascii") as fh:
                    core = fh.read().strip()
            except OSError:
                return None
            key = "pair %s/%s" % (package, core)
        cores.add(key)
    return len(cores) or None


def physical_cores():
    """`(count, how)`: the physical cores this process may run on, or `(None, why)`.

    Linux: the distinct physical cores (`_sysfs_physical_cores`) behind the CPUs in
    `sched_getaffinity(0)`, so a container or a `taskset` mask is respected and two
    hyperthreads of one core count once. Windows: every physical core of the machine.
    Anywhere else the count is unknown and nothing is capped.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
        except OSError:
            cpus = []
        if cpus:
            n = _sysfs_physical_cores(cpus)
            if n is None:
                return len(cpus), "%d CPUs in the affinity mask (no sysfs topology)" % len(cpus)
            return n, "%d physical cores under the affinity mask of %d CPUs" % (n, len(cpus))
    if sys.platform == "win32":
        n = _windows_physical_cores()
        if n:
            return n, "%d physical cores (GetLogicalProcessorInformationEx)" % n
    return None, "physical core count unknown on this platform"


def deeplc_thread_cap():
    """`(cap, why)` for DeepLC's torch CPU threads; a cap of 0 means none.

    Measured on doxy (EPYC, 64 cores, 128 CPUs): the multi-head step took 10:41 at 96
    requested threads and 18:09 at 128, because every OpenMP-parallel op waits for its
    slowest thread and a second hyperthread on a busy core is the slowest one. The cap is a
    ceiling on what the engine asks for, never a target: a request at or below it is
    taken as given. The engine asks for every logical CPU unless `--threads` says
    otherwise, so on an SMT host the cap binds by default. `MUMDIA_DEEPLC_THREAD_CAP` sets
    it explicitly; 0 disables it. A value that is not a finite number (`twelve`, `nan`,
    `inf`) is reported and ignored.
    """
    raw = os.environ.get("MUMDIA_DEEPLC_THREAD_CAP", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            value = int(float(raw))
        except (ValueError, OverflowError):
            print("WARNING: MUMDIA_DEEPLC_THREAD_CAP=%r is not a number, 0 or auto; "
                  "using auto" % raw, flush=True)
        else:
            if value <= 0:
                return 0, "MUMDIA_DEEPLC_THREAD_CAP=0 (no cap)"
            return value, "MUMDIA_DEEPLC_THREAD_CAP"
    n, why = physical_cores()
    return (n or 0), why


def capped_threads(requested, cap):
    """`requested` bounded by `cap` (0 = no cap), and at least 1."""
    requested = max(1, int(requested))
    return requested if cap <= 0 else max(1, min(requested, cap))


def is_std(pf):
    # Same predicate as `all(c in STD for c in strip_mods(base_pf(pf)))`, but the regex
    # substitution only runs when there is actually a bracketed modification to strip.
    # This is called once per library row (tens of millions), and most rows are unmodified.
    b = base_pf(pf)
    if "[" in b:
        b = strip_mods(b)
    return not (set(b) - STD)


def agg(a):
    a = np.asarray(a, dtype=np.float64)
    return a.mean(axis=1) if a.ndim == 2 else a


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


def library_bases(peptidoforms):
    """The DECOY_-stripped sequence of every library row, as `base_pf` gives it."""
    return pc.replace_substring_regex(peptidoforms, pattern=DECOY_PREFIX_RE, replacement="")


def unique_standard_bases(bases):
    """Unique stripped sequences in first-occurrence order, standard residues only.

    What the per-row loop `if b not in seen and is_std(pf)` produced, computed in Arrow so
    a library of 1e7 to 1e8 rows does not become that many Python strings and a set. The
    per-chunk uniques are widened to large_string before they are joined: at 1e8 rows they
    can pass the 2 GiB offset limit of `string`.
    """
    parts = [pc.unique(chunk).cast(pa.large_string()) for chunk in bases.chunks]
    if not parts:
        return pa.array([], pa.large_string())
    uniq = parts[0] if len(parts) == 1 else pc.unique(pa.concat_arrays(parts))
    stripped = pc.replace_substring_regex(uniq, pattern=MOD_RE, replacement="")
    return uniq.filter(pc.match_substring_regex(stripped, pattern=STD_FULL_RE))


def build_reference(args):
    """The run's confident seed PSMs as a `PSMList`: peptidoform plus observed RT.

    Shared by the fine-tune and the multi-head calibration, so both adapt to exactly the
    same peptides and both honour the held-out window rule.
    """
    # Held-out window sizing: the same rule as rt_im_train.rs::is_holdout, on the same
    # base_peptide_id, so the peptides rt-im-train scores as held-out never enter the
    # fine-tune. Pin the shared contract with the exact cases the Rust unit test uses.
    hf = args.window_holdout_frac
    if not (0.0 <= hf <= 0.9):
        raise SystemExit(f"--window-holdout-frac must be in [0.0, 0.9], got {hf}")
    is_holdout = lambda bid: bid % 1000 < round(hf * 1000)
    if hf > 0.0:
        assert [299 % 1000 < round(0.3 * 1000), 300 % 1000 < round(0.3 * 1000)] == [True, False]
    # reference: confident target seed PSMs (peptidoform + observed RT, seconds). Only the
    # columns the rule reads, and only the confident targets become Python objects: the
    # label and q filter runs in Arrow and keeps row order, so the dict below is built from
    # the same rows in the same order as a loop over the whole table.
    cols = ["peptidoform", "label", "spectrum_q", "observed_rt"]
    if hf > 0.0:
        cols.append("base_peptide_id")
    seed = pq.read_table(args.seed_path, columns=cols)
    confident = pc.and_(
        pc.equal(seed.column("label"), "target"),
        pc.less_equal(seed.column("spectrum_q").cast(pa.float64()),
                      pa.scalar(float(args.q_train), pa.float64())),
    )
    seed = seed.filter(confident).to_pydict()
    ref = {}
    n_held = 0
    for i in range(len(seed["peptidoform"])):
        pf = seed["peptidoform"][i]
        if is_std(pf):
            if hf > 0.0 and is_holdout(seed["base_peptide_id"][i]):
                n_held += 1
                continue
            ref[pf] = seed["observed_rt"][i]
    if hf > 0.0:
        print(f"window holdout: excluded {n_held} confident seed rows "
              f"(base_peptide_id %% 1000 < {round(hf * 1000)}) from the fine-tune reference", flush=True)
    ref_items = list(ref.items())
    if args.max_ref and len(ref_items) > args.max_ref:
        ref_items = ref_items[: args.max_ref]
    ref_psms = PSMList(psm_list=[PSM(peptidoform=pf, retention_time=rt, spectrum_id=str(k))
                                 for k, (pf, rt) in enumerate(ref_items)])
    print(f"reference: {len(ref_psms)} confident seed peptides", flush=True)
    return ref_psms


def fit_multihead(args, ref_psms, model=None):
    """Fit a multi-head ridge calibration of the base model against this run.

    `deeplc.predict` returns ONE of the model's 6,543 LC-setup heads, the one named by
    `DEFAULT_TASK_NAME`, on that setup's own gradient. Calibrating it with a smooth
    increasing curve, which is what the engine's LOESS does, can stretch and bend the axis
    but cannot reorder two peptides, and different chromatography reorders peptides. So the
    ordering of one arbitrary setup survives into the result no matter how good the curve
    is. `MultiHeadRidgeCalibration` ranks every head against this run's own anchors,
    spline-calibrates the best ones and ridge-combines them, so the ordering is assembled
    from the setups that actually resemble the run. It never fits more head weights than
    half the reference, so a small anchor set degrades to fewer heads rather than
    overfitting.
    """
    from deeplc.calibration import MultiHeadRidgeCalibration

    cal = MultiHeadRidgeCalibration(n_heads=args.multihead)
    print(f"multi-head calibration: fitting up to {args.multihead} heads against "
          f"{len(ref_psms)} anchors", flush=True)
    t0 = time.time()
    with quiet_deeplc_progress():
        cal = deeplc.calibrate(psm_list_reference=ref_psms, calibration=cal, model=model)
    idx = getattr(cal, "_head_idx", None)
    n_fitted = 0 if idx is None else len(idx)
    print(f"multi-head calibration: {n_fitted} heads combined, best head "
          f"{getattr(cal, 'selected_model_head', '?')}, {time.time() - t0:.1f}s", flush=True)
    return cal


def build_finetuned_model(args, ref_psms, threads):
    """Transfer-learn on the confident seed PSMs with `threads` torch CPU threads."""
    # Batch size: 0 -> auto-scale so each epoch runs ~30+ gradient steps. A fixed 512
    # underfits small references (e.g. ~4k E.coli seed = ~8 steps/epoch, never
    # converges); clamp to [16, 512].
    batch = args.batch
    if batch <= 0:
        batch = int(min(512, max(16, len(ref_psms) // 30)))
    print(f"fine-tune batch_size={batch} (~{max(1, len(ref_psms) // max(1, batch))} steps/epoch)", flush=True)

    train_kwargs = {
        "num_workers": 0,          # no DataLoader subprocesses
        "epochs": args.epochs,
        "batch_size": batch,
        "patience": args.patience,
        "device": args.device,
    }
    if args.device == "cpu":
        train_kwargs["num_threads"] = max(1, threads)   # cpu-only knob; absent in some deeplc builds
    ft_model = deeplc.finetune(ref_psms, train_kwargs=train_kwargs)   # <-- transfer learning
    print("fine-tuned model ready", flush=True)
    return ft_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib_in")
    ap.add_argument("seed_path")
    ap.add_argument("lib_out")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda moves training/inference onto the GPU; sidesteps the CPU OpenMP crash entirely")
    ap.add_argument("--threads", type=int, default=int(_THREADS),
                    help="torch CPU threads for training (bounded to avoid OpenMP oversubscription; cpu only). "
                         "Like --predict-threads it is capped at the physical cores available to the "
                         "process; MUMDIA_DEEPLC_THREAD_CAP sets the cap, 0 disables it")
    ap.add_argument("--predict-threads", type=int, default=0,
                    help="torch CPU threads for the whole-library prediction phase; "
                         "0 (default) reuses --threads, i.e. no change in behaviour. The "
                         "documented crash was OpenMP oversubscription during fine-tuning's "
                         "sustained BACKWARD pass; prediction is forward-only, and it is the "
                         "phase that dominates wall clock on a large library, so it can "
                         "usually take more threads. Raise it deliberately and watch the "
                         "per-chunk rate logged below. Capped at the physical cores "
                         "available to the process (MUMDIA_DEEPLC_THREAD_CAP, 0 = no cap): "
                         "past that, a second hyperthread on a busy core slows every "
                         "OpenMP-parallel op down (10:41 at 96 threads, 18:09 at 128 on a "
                         "64-core host).")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=0,
                    help="fine-tune batch size; 0 (default) auto-scales to the reference "
                         "size so every epoch has >= ~30 gradient steps. A fixed large "
                         "batch (e.g. 512) underfits small seeds: a ~4k-peptide E.coli "
                         "reference gives only ~8 steps/epoch and never converges.")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--q-train", dest="q_train", type=float, default=0.01,
                    help="max spectrum_q for a seed PSM to enter the fine-tune reference set")
    ap.add_argument("--max-ref", type=int, default=0,
                    help="cap reference PSMs (0 = all); use a small value for a smoke test")
    ap.add_argument("--window-holdout-frac", dest="window_holdout_frac", type=float, default=0.0,
                    help="exclude anchor peptides with base_peptide_id %% 1000 < round(frac*1000) "
                         "from the fine-tune reference. MUST match rt_im_train.window_holdout_frac: "
                         "rt-im-train sizes the RT window on exactly these held-out peptides, and "
                         "fine-tuning on them would leak adapter memorization into the residuals "
                         "(the rule is duplicated in rt_im_train.rs is_holdout; keep in sync)")
    ap.add_argument("--predict-limit", type=int, default=0,
                    help="cap number of unique peptidoforms predicted (0 = all)")
    ap.add_argument("--skip-predict", action="store_true",
                    help="fine-tune only, skip the full-library prediction (crash-path smoke test)")
    ap.add_argument("--multihead", type=int, default=0, metavar="N",
                    help="instead of fine-tuning, calibrate the base model against the run's "
                         "confident seed PSMs with MultiHeadRidgeCalibration over its N "
                         "best-correlating LC-setup heads (DeepLC >= 4.4.0). 0 (default) is "
                         "off. The engine's LOESS cannot reorder peptides, so a single head "
                         "fixes the gradient but keeps that setup's elution order; this "
                         "assembles the order from the setups that resemble the run.")
    ap.add_argument("--no-finetune", action="store_true",
                    help="skip the transfer learning and predict every peptidoform with the "
                         "DeepLC base model; seed_psms is ignored (pass '-'). The engine uses "
                         "this for rt_im_train.library_irt = deeplc, replacing an imported "
                         "library's iRT with predictions that per-run calibration then maps "
                         "onto observed RT")
    ap.add_argument("--shards", type=int, default=1, metavar="K",
                    help="split the whole-library prediction across K child processes, "
                         "with the calibration or fine-tuned model fitted once here and "
                         "handed to each. 1 (default) predicts in this process; 0 is "
                         "automatic, one shard per %d threads of the prediction thread "
                         "budget. The budget (--predict-threads after the thread cap) is "
                         "divided evenly, so K shards of budget/K threads each. Featurisation "
                         "is single-threaded Python, which is why more processes help where "
                         "more threads do not. Each child holds its own copy of the model "
                         "(0.3-0.5 GB). One process when a GPU is available." % SHARD_AUTO_THREADS)
    ap.add_argument("--predict-chunk", type=int, default=PREDICT_CHUNK, metavar="N",
                    help="unique peptidoforms per prediction call (default %d). Shards are "
                         "cut at multiples of it. Changing it changes how DeepLC batches the "
                         "sequences, so it is a test knob, not a tuning one." % PREDICT_CHUNK)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed numpy and torch before fine-tuning, so two runs on the same "
                         "input draw the same weights. Unseeded, the draw varies enough to "
                         "change results: on the AIF benchmark the held-out RT window p95 "
                         "varied 150-211 s across two draws of one arm, worth about 2 percent "
                         "of peptides, which made single-run comparisons of window sizing or "
                         "library variants unreadable. Kernel-level nondeterminism remains, "
                         "so this narrows the variance rather than removing it.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False "
                         "(wrong env? need a +cuXXX torch build)")
    # Seed before anything touches an RNG: DeepLC's transfer learning shuffles and
    # initialises, and an unseeded draw is the largest source of run-to-run
    # variation in the whole pipeline (docs/14_build_test_deploy_gotchas.md).
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed={args.seed} (numpy, torch"
          f"{', cuda' if torch.cuda.is_available() else ''}); "
          "training kernels are not guaranteed bit-for-bit deterministic", flush=True)

    # Bound torch's own thread pool; only one OpenMP pool spins now (matters on cpu). Both
    # the training and the prediction pool are capped at the physical cores this process
    # may use (`deeplc_thread_cap`); a request at or below the cap is taken as given.
    cap, cap_why = deeplc_thread_cap()
    train_threads = capped_threads(args.threads, cap)
    predict_asked = args.predict_threads if args.predict_threads > 0 else args.threads
    predict_threads = capped_threads(predict_asked, cap)
    thread_record = {
        "requested_train": max(1, args.threads),
        "requested_predict": max(1, predict_asked),
        "cap": cap,
        "cap_source": cap_why,
        "train": train_threads,
        "predict": predict_threads,
    }
    torch.set_num_threads(train_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized
    if args.device == "cuda":
        print(f"device=cuda gpu={torch.cuda.get_device_name(0)}; compute off the CPU OpenMP pools", flush=True)
    else:
        print(f"device=cpu torch threads={torch.get_num_threads()} interop=1; OMP/BLAS pinned to 1", flush=True)
    print(f"torch thread cap {cap if cap > 0 else 'none'} ({cap_why}): training "
          f"{train_threads} of {thread_record['requested_train']} asked, prediction "
          f"{predict_threads} of {thread_record['requested_predict']} asked", flush=True)
    if (train_threads < thread_record["requested_train"]
            or predict_threads < thread_record["requested_predict"]):
        print("the thread cap lowered the requested torch threads; set "
              "MUMDIA_DEEPLC_THREAD_CAP=0 to take the request as given", flush=True)

    # Wall time per phase, in seconds, for the summary (survey P0). None where a phase did
    # not run or could not be measured.
    timings = {}
    t_phase = time.perf_counter()
    lib = pq.read_table(args.lib_in)
    orig = np.asarray(lib.column("predicted_irt"), dtype=np.float32)
    timings["read_library"] = round(time.perf_counter() - t_phase, 3)

    if args.multihead and args.no_finetune:
        raise SystemExit("--multihead and --no-finetune are alternatives: the first "
                         "calibrates the base model against this run, the second predicts "
                         "from it uncalibrated")
    if args.multihead < 0:
        raise SystemExit(f"--multihead must be >= 0, got {args.multihead}")

    calibration = None
    base_model = None
    timings["model_load"] = None
    timings["reference"] = None
    timings["fit"] = None
    if args.multihead:
        t_phase = time.perf_counter()
        base_model = load_base_model()
        timings["model_load"] = round(time.perf_counter() - t_phase, 3)
        t_phase = time.perf_counter()
        ref_psms = build_reference(args)
        timings["reference"] = round(time.perf_counter() - t_phase, 3)
        t_phase = time.perf_counter()
        calibration = fit_multihead(args, ref_psms, model=base_model)
        timings["fit"] = round(time.perf_counter() - t_phase, 3)
        ft_model = None
    elif args.no_finetune:
        ft_model = None
        print("no-finetune: predicting with the DeepLC base model (seed ignored)", flush=True)
    else:
        t_phase = time.perf_counter()
        ref_psms = build_reference(args)
        timings["reference"] = round(time.perf_counter() - t_phase, 3)
        # The fine-tune loads its own copy of the model inside `deeplc.finetune`, so that
        # load is part of "fit" here.
        t_phase = time.perf_counter()
        ft_model = build_finetuned_model(args, ref_psms, train_threads)
        timings["fit"] = round(time.perf_counter() - t_phase, 3)

    if args.skip_predict:
        print("skip-predict set; fine-tune smoke test complete (crash path exercised)", flush=True)
        return

    # Deduplicate and predict on the DECOY_-stripped underlying sequence so decoys are
    # fine-tuned onto the same iRT scale as targets (shift-decoys reuse their target's
    # prediction; reverse-decoys get their reversed-sequence prediction).
    t_phase = time.perf_counter()
    bases = library_bases(lib.column("peptidoform"))
    uniq = unique_standard_bases(bases)
    if args.predict_limit:
        uniq = uniq.slice(0, args.predict_limit)
    timings["unique"] = round(time.perf_counter() - t_phase, 3)
    if args.predict_chunk < 1:
        raise SystemExit(f"--predict-chunk must be >= 1, got {args.predict_chunk}")
    chunk = args.predict_chunk
    n_shards, shard_threads, shard_note = shard_plan(
        args.shards, predict_threads, len(uniq), chunk, torch.cuda.is_available())
    which = (
        f"the base model calibrated over {args.multihead} heads"
        if calibration is not None
        else "the DeepLC base model"
        if ft_model is None
        else "the fine-tuned model"
    )
    shard_record = {
        "requested": args.shards,
        "used": n_shards,
        "threads_per_shard": shard_threads,
        "chunk": chunk,
        "plan": shard_note,
    }
    t_pred0 = time.time()
    if n_shards == 1:
        if ft_model is None and base_model is None:
            t_phase = time.perf_counter()
            base_model = load_base_model()
            timings["model_load"] = round(time.perf_counter() - t_phase, 3)
        model = ft_model if ft_model is not None else base_model
        if shard_threads != torch.get_num_threads():
            torch.set_num_threads(shard_threads)
        print(f"predicting {len(uniq)} unique standard peptidoforms with {which} "
              f"(torch threads={torch.get_num_threads()})", flush=True)
        values, timers = predict_values(uniq, model, calibration, chunk)
        timings["featurisation"], timings["forward"] = timers.totals()
    else:
        print(f"predicting {len(uniq)} unique standard peptidoforms with {which} in "
              f"{n_shards} processes of {shard_threads} torch threads ({shard_note})",
              flush=True)
        values, per_shard = predict_sharded(
            uniq, n_shards, shard_threads, chunk, ft_model, calibration, args.lib_out)
        shard_record["per_shard"] = per_shard
        # Summed over the shards, so these are process-seconds rather than wall time.
        for key in ("featurisation", "forward"):
            parts = [p.get(key) for p in per_shard]
            timings[key] = None if any(v is None for v in parts) else round(sum(parts), 3)
    timings["predict"] = round(time.time() - t_pred0, 3)
    print(f"prediction phase: {time.time() - t_pred0:.1f}s total "
          f"(featurisation {_fmt_s(timings['featurisation'])}, "
          f"forward pass {_fmt_s(timings['forward'])}"
          f"{', summed over shards' if n_shards > 1 else ''})", flush=True)

    t_phase = time.perf_counter()
    new, summary = rewrite_irt(bases, orig, uniq, values)
    timings["rewrite"] = round(time.perf_counter() - t_phase, 3)
    t_phase = time.perf_counter()
    idx = lib.schema.get_field_index("predicted_irt")
    lib = lib.set_column(idx, "predicted_irt", pa.array(new, pa.float32()))
    pq.write_table(lib, args.lib_out)
    timings["write"] = round(time.perf_counter() - t_phase, 3)
    summary["model"] = which
    summary["lib_in"] = args.lib_in
    summary["lib_out"] = args.lib_out
    summary["unique_predicted"] = len(uniq)
    summary["torch_threads"] = thread_record
    summary["shards"] = shard_record
    summary["timings_s"] = timings
    if calibration is not None:
        summary["multihead"] = multihead_record(calibration, args.multihead, len(ref_psms))
    with open(args.lib_out + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote library with re-predicted iRT ({which}): {args.lib_out}")
    print(f"  rows={summary['rows']} repredicted={summary['repredicted']} "
          f"retained_imported={summary['retained_imported']} "
          f"(non-standard residues {summary['retained_non_standard']}, "
          f"no finite prediction {summary['retained_no_prediction']})")
    if summary["retained_imported"]:
        print(f"WARNING: {summary['retained_imported']} of {summary['rows']} rows "
              f"({100.0 * summary['retained_imported'] / max(1, summary['rows']):.2f}%) keep "
              f"their imported iRT, which is on the imported model's scale, not {which}'s; "
              f"the counts are in {args.lib_out}.summary.json", flush=True)


def multihead_record(calibration, requested, anchors):
    """What the multi-head fit chose, so two runs can be compared head for head.

    A thread-count or version change moves the reference predictions in the last bits, and
    the survey's validation asks whether the selected heads stayed the same. None of these
    are read back by the engine.
    """
    idx = getattr(calibration, "_head_idx", None)
    ridge = getattr(calibration, "_ridge", None)
    alpha = getattr(ridge, "alpha_", None)
    best = getattr(calibration, "selected_model_head", None)
    return {
        "heads_requested": int(requested),
        "anchors": int(anchors),
        "heads": None if idx is None else [int(h) for h in idx],
        "best_head": None if best is None else int(best),
        "ridge_alpha": None if alpha is None else float(alpha),
    }


def _fmt_s(value):
    return "n/a" if value is None else f"{value:.1f}s"


def predict_values(uniq, model, calibration, chunk, label=""):
    """Predictions (float64, aligned with `uniq`) in `chunk`-sized calls, and their timers.

    The single loop both the one-process path and every shard run, so a shard makes exactly
    the calls the one process would have made over the same slice.
    """
    # Only needed to satisfy `predict_and_calibrate`, which parses a reference before it
    # notices the calibration is already fitted.
    ref_for_transform = ref_psms_for_transform(calibration, None)
    values = np.empty(len(uniq), dtype=np.float64)
    timers = PredictTimers()
    prefix = f"[{label}] " if label else ""
    # DeepLC's progress writer emits one blank line per update when stdout is not a
    # terminal, and the engine inherits this worker's stdout, so a real run was 98%
    # blank lines. The per-chunk progress printed inside the loop is not blank and
    # still comes through.
    with quiet_deeplc_progress(), timers.install(model):
        for s in range(0, len(uniq), chunk):
            t0 = time.time()
            batch = uniq.slice(s, chunk).to_pylist()
            if calibration is not None:
                # The calibration is already fitted, so this only predicts and transforms: it
                # pulls the head columns the ridge reads rather than materialising all 6,543,
                # which at library scale would be terabytes. The reference is passed again
                # because the signature requires one; the fitting step is skipped.
                p = agg(deeplc.predict_and_calibrate(
                    batch, psm_list_reference=ref_for_transform, calibration=calibration,
                    model=model))
            else:
                p = agg(deeplc.predict(batch, model=model))
            # A structurally short or long answer is a broken predictor, not a set of
            # unsupported peptidoforms: zipping it silently paired predictions with the wrong
            # peptidoforms and left the tail on its imported value (docs/30 R6).
            if len(p) != len(batch):
                raise SystemExit(
                    f"DeepLC returned {len(p)} predictions for {len(batch)} peptidoforms in one "
                    f"batch; refusing to rewrite the library from a malformed response")
            values[s:s + len(batch)] = p
            done = min(s + chunk, len(uniq))
            dt = time.time() - t0
            rate = len(batch) / dt if dt > 0 else float("inf")
            eta = (len(uniq) - done) / rate if rate > 0 else float("nan")
            print(f"  {prefix}{done}/{len(uniq)}  {dt:.1f}s for this chunk "
                  f"({rate:.0f} peptidoforms/s, ETA {eta / 60:.1f} min)", flush=True)
    return values, timers


def shard_plan(requested, budget, n_items, chunk, gpu):
    """`(shards, threads per shard, why)` for the whole-library prediction.

    A deterministic function of the request, the thread budget (the prediction threads
    after the cap) and the library, never of free memory, so two runs of one configuration
    on one host split the same way. K = `requested`, or budget / SHARD_AUTO_THREADS when it
    is 0, bounded by the budget (at least one thread per shard); each shard gets budget / K
    threads, rounded down. The slices are whole chunks, so a library of fewer chunks than K
    uses fewer shards at the same threads each, and a library of one chunk is predicted in
    this process with the whole budget. A GPU is one device, so it always gets one process.
    """
    if requested < 0:
        raise SystemExit(f"--shards must be >= 0, got {requested}")
    budget = max(1, int(budget))
    if gpu:
        return 1, budget, "a GPU is available, so one process"
    k = requested if requested > 0 else max(1, budget // SHARD_AUTO_THREADS)
    k = max(1, min(k, budget))
    if k == 1:
        return 1, budget, "one process"
    threads = max(1, budget // k)
    n_chunks = max(1, -(-n_items // chunk))
    per = -(-n_chunks // min(k, n_chunks))
    used = -(-n_chunks // per)
    if used == 1:
        return 1, budget, f"{n_chunks} chunk(s) of {chunk}, so one process"
    why = f"{k} requested" if requested > 0 else f"automatic: budget {budget} / {SHARD_AUTO_THREADS}"
    if used < k:
        why += f", {used} used for {n_chunks} chunks of {chunk}"
    return used, threads, why


def shard_bounds(n_items, shards, chunk):
    """Contiguous `(start, stop)` slices of `n_items`, every start a multiple of `chunk`."""
    n_chunks = max(1, -(-n_items // chunk))
    per = -(-n_chunks // shards) * chunk
    return [(a, min(a + per, n_items)) for a in range(0, n_items, per)]


def predict_sharded(uniq, shards, threads, chunk, ft_model, calibration, lib_out):
    """Predict `uniq` in `shards` child processes and join the slices in order.

    The fitted calibration (pickled) or fine-tuned model (`torch.save` of the module) is
    written once and every child reads it, so no shard refits: a refit could select a
    different head at rank 80 from last-bit differences in the reference predictions. Each
    child's slice starts at a multiple of `chunk` and it predicts in `chunk`-sized calls
    from there, which are the calls the one process would have made. A child that exits
    non-zero stops the others and fails the stage; nothing is written then. The scratch
    directory beside `lib_out` is removed either way.
    """
    bounds = shard_bounds(len(uniq), shards, chunk)
    base_dir = os.path.dirname(os.path.abspath(lib_out))
    work = tempfile.mkdtemp(prefix=os.path.basename(lib_out) + ".shards.", dir=base_dir)
    procs = []
    try:
        cal_path = model_path = None
        if calibration is not None:
            cal_path = os.path.join(work, "calibration.pkl")
            with open(cal_path, "wb") as fh:
                pickle.dump(calibration, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if ft_model is not None:
            model_path = os.path.join(work, "model.pt")
            torch.save(ft_model, model_path)
        specs = []
        for j, (a, b) in enumerate(bounds):
            stem = os.path.join(work, f"shard_{j:03d}")
            pq.write_table(pa.table({"seq": uniq.slice(a, b - a)}), stem + "_seqs.parquet")
            spec = {
                "shard": j,
                "label": f"shard {j + 1}/{len(bounds)}",
                "seqs": stem + "_seqs.parquet",
                "out": stem + "_values.npy",
                "timings": stem + "_timings.json",
                "threads": threads,
                "chunk": chunk,
                "calibration": cal_path,
                "model": model_path,
                "rows": b - a,
            }
            with open(stem + "_spec.json", "w", encoding="utf-8") as fh:
                json.dump(spec, fh)
            specs.append(spec)
        script = os.path.abspath(__file__)
        for j, spec in enumerate(specs):
            spec_path = os.path.join(work, f"shard_{j:03d}_spec.json")
            procs.append(subprocess.Popen([sys.executable, script, "--shard-worker", spec_path]))
        _wait_for_shards(procs)
        parts, per_shard = [], []
        for spec in specs:
            v = np.load(spec["out"])
            if v.shape != (spec["rows"],):
                raise SystemExit(
                    f"prediction {spec['label']} returned {v.shape[0]} values for "
                    f"{spec['rows']} sequences; refusing to rewrite the library")
            parts.append(v)
            with open(spec["timings"], encoding="utf-8") as fh:
                per_shard.append(json.load(fh))
        values = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        shutil.rmtree(work, ignore_errors=True)
    return values, per_shard


def _wait_for_shards(procs):
    """Wait for every shard; on the first failure stop the rest and fail the stage."""
    while True:
        alive = 0
        for j, proc in enumerate(procs):
            rc = proc.poll()
            if rc is None:
                alive += 1
            elif rc != 0:
                for other in procs:
                    if other.poll() is None:
                        other.kill()
                for other in procs:
                    other.wait()
                raise SystemExit(
                    f"prediction shard {j + 1}/{len(procs)} exited with status {rc}; its "
                    f"output is above. The other shards were stopped and no library was "
                    f"written.")
        if alive == 0:
            return
        time.sleep(0.25)


def shard_main(spec_path):
    """One prediction shard (`--shard-worker <spec.json>`), started by `predict_sharded`."""
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    torch.set_num_threads(max(1, int(spec["threads"])))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.perf_counter()
    if spec.get("model"):
        # A module saved whole by the parent from its own fine-tune; shards run only
        # without a GPU (`shard_plan`), so it is mapped onto the CPU.
        model = torch.load(spec["model"], map_location="cpu", weights_only=False)
    else:
        model = load_base_model()
    t_load = time.perf_counter() - t0
    calibration = None
    if spec.get("calibration"):
        with open(spec["calibration"], "rb") as fh:
            calibration = pickle.load(fh)
    uniq = pq.read_table(spec["seqs"]).column("seq").combine_chunks()
    t1 = time.perf_counter()
    values, timers = predict_values(uniq, model, calibration, int(spec["chunk"]),
                                    label=spec["label"])
    t_predict = time.perf_counter() - t1
    np.save(spec["out"], values)
    featurisation, forward = timers.totals()
    with open(spec["timings"], "w", encoding="utf-8") as fh:
        json.dump({
            "shard": spec["shard"],
            "rows": len(uniq),
            "threads": torch.get_num_threads(),
            "model_load": round(t_load, 3),
            "predict": round(t_predict, 3),
            "featurisation": featurisation,
            "forward": forward,
        }, fh)


def ref_psms_for_transform(calibration, args):
    """A one-row reference for the transform-only calls, or None when not calibrating."""
    if calibration is None:
        return None
    return PSMList(psm_list=[PSM(peptidoform="PEPTIDEK", retention_time=0.0, spectrum_id="0")])


def rewrite_irt(bases, orig, uniq, values):
    """The new `predicted_irt` column and a count of where each value came from.

    `bases` is the DECOY_-stripped sequence of every library row, `uniq` the sequences
    that were predicted and `values` their predictions (float64, aligned with `uniq`). A
    row whose sequence has a finite prediction takes it, rounded to float32. A row without
    one keeps its imported value: rows with non-standard residues are never sent to DeepLC
    (`is_std`), rows past `--predict-limit` were not predicted, and a prediction that came
    back non-finite is an unsupported input rather than a number. Both kinds are counted so
    the mixture of RT sources in the written library is explicit instead of silent (docs/30
    R6). The counts and the column are what the per-row dictionary lookup gave; only the
    lookup moved into Arrow (`index_in`), so a 1e7-row library is not walked in Python.
    """
    n = len(bases)
    probe, value_set = bases, uniq
    try:
        # `index_in` needs one string type on both sides. The library column is usually
        # `string`; the unique set is large_string (see `unique_standard_bases`) and is
        # narrowed when it fits, else the probe is widened.
        value_set = uniq.cast(bases.type)
    except (pa.ArrowInvalid, pa.ArrowCapacityError, OverflowError):
        probe = bases.cast(pa.large_string())
    pos = pc.fill_null(pc.index_in(probe, value_set=value_set), -1)
    pos = np.asarray(pos.to_numpy(), dtype=np.int64)
    found = pos >= 0
    vals = np.full(n, np.nan, dtype=np.float64)
    vals[found] = values[pos[found]]
    good = np.isfinite(vals)  # NaN where not found, so this is found AND finite
    new = orig.astype(np.float32, copy=True)
    new[good] = vals[good].astype(np.float32)
    repredicted = int(good.sum())
    non_standard = int(n - found.sum())
    no_prediction = int(found.sum()) - repredicted
    return new, {
        "rows": n,
        "repredicted": repredicted,
        "retained_imported": non_standard + no_prediction,
        "retained_non_standard": non_standard,
        "retained_no_prediction": no_prediction,
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--shard-worker":
        shard_main(sys.argv[2])
    else:
        main()
