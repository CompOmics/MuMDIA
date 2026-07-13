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
import deeplc                                    # import before numpy (OpenMP load order)
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
is_std = lambda pf: all(c in STD for c in strip_mods(base_pf(pf)))


def agg(a):
    a = np.asarray(a, dtype=np.float64)
    return a.mean(axis=1) if a.ndim == 2 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib_in")
    ap.add_argument("seed_path")
    ap.add_argument("lib_out")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda moves training/inference onto the GPU; sidesteps the CPU OpenMP crash entirely")
    ap.add_argument("--threads", type=int, default=int(_THREADS),
                    help="torch CPU threads for training (bounded to avoid OpenMP oversubscription; cpu only)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-ref", type=int, default=0,
                    help="cap reference PSMs (0 = all); use a small value for a smoke test")
    ap.add_argument("--predict-limit", type=int, default=0,
                    help="cap number of unique peptidoforms predicted (0 = all)")
    ap.add_argument("--skip-predict", action="store_true",
                    help="fine-tune only, skip the full-library prediction (crash-path smoke test)")
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

    # reference: confident target seed PSMs (peptidoform + observed RT, seconds)
    seed = pq.read_table(args.seed_path).to_pydict()
    ref = {}
    for i in range(len(seed["peptidoform"])):
        pf = seed["peptidoform"][i]
        if seed["label"][i] == "target" and seed["spectrum_q"][i] <= 0.01 and is_std(pf):
            ref[pf] = seed["observed_rt"][i]
    ref_items = list(ref.items())
    if args.max_ref and len(ref_items) > args.max_ref:
        ref_items = ref_items[: args.max_ref]
    ref_psms = PSMList(psm_list=[PSM(peptidoform=pf, retention_time=rt, spectrum_id=str(k))
                                 for k, (pf, rt) in enumerate(ref_items)])
    print(f"fine-tune reference: {len(ref_psms)} confident seed peptides", flush=True)

    train_kwargs = {
        "num_workers": 0,          # no DataLoader subprocesses
        "epochs": args.epochs,
        "batch_size": args.batch,
        "patience": args.patience,
        "device": args.device,
    }
    if args.device == "cpu":
        train_kwargs["num_threads"] = max(1, args.threads)   # cpu-only knob; absent in some deeplc builds
    ft_model = deeplc.finetune(ref_psms, train_kwargs=train_kwargs)   # <-- transfer learning
    print("fine-tuned model ready", flush=True)

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
    print(f"predicting {len(uniq)} unique standard peptidoforms with fine-tuned model", flush=True)
    preds = {}
    chunk = 100_000
    for s in range(0, len(uniq), chunk):
        batch = uniq[s:s + chunk]
        p = agg(deeplc.predict(batch, model=ft_model))
        for pf, v in zip(batch, p):
            preds[pf] = float(v)
        print(f"  {min(s + chunk, len(uniq))}/{len(uniq)}", flush=True)

    new = np.array([preds.get(base_pf(pf), orig[i]) for i, pf in enumerate(pform)], dtype=np.float32)
    idx = lib.schema.get_field_index("predicted_irt")
    lib = lib.set_column(idx, "predicted_irt", pa.array(new, pa.float32()))
    pq.write_table(lib, args.lib_out)
    print(f"wrote fine-tuned library: {args.lib_out}")


if __name__ == "__main__":
    main()
