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
            "deeplc %s is older than the required %d.%d.%d (pip install 'deeplc>=4.1.1')"
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


def build_finetuned_model(args):
    """Transfer-learn on the confident seed PSMs named by args.seed_path."""
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
    print(f"fine-tune reference: {len(ref_psms)} confident seed peptides", flush=True)

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
        train_kwargs["num_threads"] = max(1, args.threads)   # cpu-only knob; absent in some deeplc builds
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
                    help="torch CPU threads for training (bounded to avoid OpenMP oversubscription; cpu only)")
    ap.add_argument("--predict-threads", type=int, default=0,
                    help="torch CPU threads for the whole-library prediction phase; "
                         "0 (default) reuses --threads, i.e. no change in behaviour. The "
                         "documented crash was OpenMP oversubscription during fine-tuning's "
                         "sustained BACKWARD pass; prediction is forward-only, and it is the "
                         "phase that dominates wall clock on a large library, so it can "
                         "usually take more threads. Raise it deliberately and watch the "
                         "per-chunk rate logged below.")
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
    ap.add_argument("--no-finetune", action="store_true",
                    help="skip the transfer learning and predict every peptidoform with the "
                         "DeepLC base model; seed_psms is ignored (pass '-'). The engine uses "
                         "this for rt_im_train.library_irt = deeplc, replacing an imported "
                         "library's iRT with predictions that per-run calibration then maps "
                         "onto observed RT")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False "
                         "(wrong env? need a +cuXXX torch build)")
    # bound torch's own thread pool; only one OpenMP pool spins now (matters on cpu)
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized
    if args.device == "cuda":
        print(f"device=cuda gpu={torch.cuda.get_device_name(0)}; compute off the CPU OpenMP pools", flush=True)
    else:
        print(f"device=cpu torch threads={torch.get_num_threads()} interop=1; OMP/BLAS pinned to 1", flush=True)

    lib = pq.read_table(args.lib_in)
    pform = lib.column("peptidoform").to_pylist()
    orig = np.asarray(lib.column("predicted_irt"), dtype=np.float32)

    if args.no_finetune:
        ft_model = None
        print("no-finetune: predicting with the DeepLC base model (seed ignored)", flush=True)
    else:
        ft_model = build_finetuned_model(args)

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
    pt = args.predict_threads if args.predict_threads > 0 else args.threads
    if pt != torch.get_num_threads():
        torch.set_num_threads(max(1, pt))
    which = "the DeepLC base model" if ft_model is None else "the fine-tuned model"
    print(f"predicting {len(uniq)} unique standard peptidoforms with {which} "
          f"(torch threads={torch.get_num_threads()})", flush=True)
    preds = {}
    chunk = 100_000
    t_pred0 = time.time()
    for s in range(0, len(uniq), chunk):
        t0 = time.time()
        batch = uniq[s:s + chunk]
        p = agg(deeplc.predict(batch) if ft_model is None else deeplc.predict(batch, model=ft_model))
        for pf, v in zip(batch, p):
            preds[pf] = float(v)
        done = min(s + chunk, len(uniq))
        dt = time.time() - t0
        rate = len(batch) / dt if dt > 0 else float("inf")
        eta = (len(uniq) - done) / rate if rate > 0 else float("nan")
        print(f"  {done}/{len(uniq)}  {dt:.1f}s for this chunk "
              f"({rate:.0f} peptidoforms/s, ETA {eta / 60:.1f} min)", flush=True)
    print(f"prediction phase: {time.time() - t_pred0:.1f}s total", flush=True)

    # `base_pf` is recomputed here rather than cached from the pass above on purpose:
    # caching it would retain one extra string per library row (hundreds of MB at
    # library scale) to avoid a `startswith` and a slice.
    new = np.array([preds.get(base_pf(pf), orig[i]) for i, pf in enumerate(pform)], dtype=np.float32)
    idx = lib.schema.get_field_index("predicted_irt")
    lib = lib.set_column(idx, "predicted_irt", pa.array(new, pa.float32()))
    pq.write_table(lib, args.lib_out)
    print(f"wrote library with re-predicted iRT ({which}): {args.lib_out}")


if __name__ == "__main__":
    main()
