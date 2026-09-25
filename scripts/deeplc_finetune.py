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
prediction) are capped at the physical cores available to the process: the unique
(package, core) pairs under the affinity mask on Linux, every physical core on Windows.
MUMDIA_DEEPLC_THREAD_CAP=N sets the cap explicitly and 0 disables it. The resolved
numbers are printed and recorded under "torch_threads" in <lib_out>.summary.json.
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
import math
import re
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
import pyarrow.parquet as pq
import torch
from psm_utils import PSM, PSMList

STD = set("ACDEFGHIKLMNPQRSTVWY")
strip_mods = lambda s: re.sub(r"\[[^\]]*\]", "", s)
# Strip the decoy marker before prediction: a "DECOY_" peptidoform must be predicted on
# its underlying sequence, else is_std rejects it (the '_') and the decoy keeps the base
# (un-fine-tuned) iRT, landing on a different scale than the fine-tuned targets.
base_pf = lambda s: s[6:] if s.startswith("DECOY_") else s



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


def physical_cores():
    """`(count, how)`: the physical cores this process may run on, or `(None, why)`.

    Linux: the unique (physical_package_id, core_id) pairs from sysfs over the CPUs in
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
            pairs = set()
            for cpu in cpus:
                topo = "/sys/devices/system/cpu/cpu%d/topology/" % cpu
                try:
                    with open(topo + "physical_package_id", encoding="ascii") as fh:
                        package = fh.read().strip()
                    with open(topo + "core_id", encoding="ascii") as fh:
                        core = fh.read().strip()
                except OSError:
                    return len(cpus), "%d CPUs in the affinity mask (no sysfs topology)" % len(cpus)
                pairs.add((package, core))
            return len(pairs), "%d physical cores under the affinity mask of %d CPUs" % (
                len(pairs), len(cpus))
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
    ceiling on what the engine asks for, never a target, so below it nothing changes.
    `MUMDIA_DEEPLC_THREAD_CAP` sets it explicitly; 0 disables it.
    """
    raw = os.environ.get("MUMDIA_DEEPLC_THREAD_CAP", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            value = int(float(raw))
        except ValueError:
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


def build_reference(args):
    """The run's confident seed PSMs as a `PSMList`: peptidoform plus observed RT.

    Shared by the fine-tune and the multi-head calibration, so both adapt to exactly the
    same peptides and both honour the held-out window rule.
    """
    # reference: confident target seed PSMs (peptidoform + observed RT, seconds)
    seed = pq.read_table(args.seed_path).to_pydict()
    # Held-out window sizing: the same rule as rt_im_train.rs::is_holdout, on the same
    # base_peptide_id, so the peptides rt-im-train scores as held-out never enter the
    # fine-tune. Pin the shared contract with the exact cases the Rust unit test uses.
    hf = args.window_holdout_frac
    if not (0.0 <= hf <= 0.9):
        raise SystemExit(f"--window-holdout-frac must be in [0.0, 0.9], got {hf}")
    is_holdout = lambda bid: bid % 1000 < round(hf * 1000)
    if hf > 0.0:
        assert [299 % 1000 < round(0.3 * 1000), 300 % 1000 < round(0.3 * 1000)] == [True, False]
    ref = {}
    n_held = 0
    for i in range(len(seed["peptidoform"])):
        pf = seed["peptidoform"][i]
        if seed["label"][i] == "target" and seed["spectrum_q"][i] <= args.q_train and is_std(pf):
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


def fit_multihead(args, ref_psms):
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
        cal = deeplc.calibrate(psm_list_reference=ref_psms, calibration=cal)
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

    lib = pq.read_table(args.lib_in)
    pform = lib.column("peptidoform").to_pylist()
    orig = np.asarray(lib.column("predicted_irt"), dtype=np.float32)

    if args.multihead and args.no_finetune:
        raise SystemExit("--multihead and --no-finetune are alternatives: the first "
                         "calibrates the base model against this run, the second predicts "
                         "from it uncalibrated")
    if args.multihead < 0:
        raise SystemExit(f"--multihead must be >= 0, got {args.multihead}")

    calibration = None
    if args.multihead:
        calibration = fit_multihead(args, build_reference(args))
        ft_model = None
    elif args.no_finetune:
        ft_model = None
        print("no-finetune: predicting with the DeepLC base model (seed ignored)", flush=True)
    else:
        ft_model = build_finetuned_model(args, build_reference(args), train_threads)

    if args.skip_predict:
        print("skip-predict set; fine-tune smoke test complete (crash path exercised)", flush=True)
        return

    # Deduplicate and predict on the DECOY_-stripped underlying sequence so decoys are
    # fine-tuned onto the same iRT scale as targets (shift-decoys reuse their target's
    # prediction; reverse-decoys get their reversed-sequence prediction).
    uniq, seen = [], set()
    for pf in pform:
        b = base_pf(pf)
        if b not in seen and is_std(pf):
            seen.add(b)
            uniq.append(b)
    if args.predict_limit:
        uniq = uniq[: args.predict_limit]
    if predict_threads != torch.get_num_threads():
        torch.set_num_threads(predict_threads)
    which = (
        f"the base model calibrated over {args.multihead} heads"
        if calibration is not None
        else "the DeepLC base model"
        if ft_model is None
        else "the fine-tuned model"
    )
    print(f"predicting {len(uniq)} unique standard peptidoforms with {which} "
          f"(torch threads={torch.get_num_threads()})", flush=True)
    # Only needed to satisfy `predict_and_calibrate`, which parses a reference before it
    # notices the calibration is already fitted.
    ref_for_transform = ref_psms_for_transform(calibration, args)
    preds = {}
    chunk = 100_000
    t_pred0 = time.time()
    # DeepLC's progress writer emits one blank line per update when stdout is not a
    # terminal, and the engine inherits this worker's stdout, so a real run was 98%
    # blank lines. The per-chunk progress printed inside the loop is not blank and
    # still comes through.
    with quiet_deeplc_progress():
        for s in range(0, len(uniq), chunk):
            t0 = time.time()
            batch = uniq[s:s + chunk]
            if calibration is not None:
                # The calibration is already fitted, so this only predicts and transforms: it
                # pulls the head columns the ridge reads rather than materialising all 6,543,
                # which at library scale would be terabytes. The reference is passed again
                # because the signature requires one; the fitting step is skipped.
                p = agg(deeplc.predict_and_calibrate(
                    batch, psm_list_reference=ref_for_transform, calibration=calibration))
            else:
                p = agg(deeplc.predict(batch) if ft_model is None else deeplc.predict(batch, model=ft_model))
            # A structurally short or long answer is a broken predictor, not a set of
            # unsupported peptidoforms: zipping it silently paired predictions with the wrong
            # peptidoforms and left the tail on its imported value (docs/30 R6).
            if len(p) != len(batch):
                raise SystemExit(
                    f"DeepLC returned {len(p)} predictions for {len(batch)} peptidoforms in one "
                    f"batch; refusing to rewrite the library from a malformed response")
            for pf, v in zip(batch, p):
                preds[pf] = float(v)
            done = min(s + chunk, len(uniq))
            dt = time.time() - t0
            rate = len(batch) / dt if dt > 0 else float("inf")
            eta = (len(uniq) - done) / rate if rate > 0 else float("nan")
            print(f"  {done}/{len(uniq)}  {dt:.1f}s for this chunk "
                  f"({rate:.0f} peptidoforms/s, ETA {eta / 60:.1f} min)", flush=True)
    print(f"prediction phase: {time.time() - t_pred0:.1f}s total", flush=True)

    new, summary = rewrite_irt(pform, orig, preds)
    idx = lib.schema.get_field_index("predicted_irt")
    lib = lib.set_column(idx, "predicted_irt", pa.array(new, pa.float32()))
    pq.write_table(lib, args.lib_out)
    summary["model"] = which
    summary["lib_in"] = args.lib_in
    summary["lib_out"] = args.lib_out
    summary["torch_threads"] = thread_record
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


def ref_psms_for_transform(calibration, args):
    """A one-row reference for the transform-only calls, or None when not calibrating."""
    if calibration is None:
        return None
    return PSMList(psm_list=[PSM(peptidoform="PEPTIDEK", retention_time=0.0, spectrum_id="0")])


def rewrite_irt(pform, orig, preds):
    """The new `predicted_irt` column and a count of where each value came from.

    A peptidoform with a finite prediction for its DECOY_-stripped sequence takes it. A
    peptidoform without one keeps its imported value: rows with non-standard residues are
    never sent to DeepLC (`is_std`), and a prediction that came back non-finite is an
    unsupported input rather than a number. Both are counted so the mixture of RT sources
    in the written library is explicit instead of silent (docs/30 R6). `base_pf` is
    recomputed here rather than cached from the pass above on purpose: caching it would
    retain one extra string per library row (hundreds of MB at library scale).
    """
    n = len(pform)
    new = np.empty(n, dtype=np.float32)
    repredicted = 0
    no_prediction = 0
    non_standard = 0
    for i, pf in enumerate(pform):
        v = preds.get(base_pf(pf))
        if v is None:
            new[i] = orig[i]
            non_standard += 1
        elif not math.isfinite(v):
            new[i] = orig[i]
            no_prediction += 1
        else:
            new[i] = v
            repredicted += 1
    return new, {
        "rows": n,
        "repredicted": repredicted,
        "retained_imported": non_standard + no_prediction,
        "retained_non_standard": non_standard,
        "retained_no_prediction": no_prediction,
    }


if __name__ == "__main__":
    main()
