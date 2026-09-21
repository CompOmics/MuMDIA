"""Sharded, deduplicated multi-head DeepLC calibration of a very large precursor table.

  python mh_shard_predict.py uniq    <lib_precursors> <out_prefix> <n_shards>
  python mh_shard_predict.py predict <shard.parquet> <seed_psms.parquet> <out_preds.parquet>
                                     [--holdout F] [--threads T] [--heads N] [--limit K]
  python mh_shard_predict.py merge   <lib_precursors> <out_precursors> <preds.parquet> [...]

Reuses deeplc_finetune.py's reference builder, multi-head fit and calibrated prediction
call (imported from the script directory, or MUMDIA_SCRIPTS), so the merged table is what one
`deeplc_finetune.py --multihead` run writes, minus its single-process wall clock: the
unique DECOY_-stripped standard sequences are predicted once across all shards. A row-range
shard of an m/z-sorted table would predict every charge state separately (2.4x the work on
the 9-mer immunopeptidomics library).
"""
import argparse
import json
import os
import sys
import time

# deeplc_finetune.py lives next to this script; MUMDIA_SCRIPTS points elsewhere when a
# different checkout of the worker should be used.
sys.path.insert(0, os.environ.get("MUMDIA_SCRIPTS", os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

STD_RE = r"^[ACDEFGHIKLMNPQRSTVWY]+$"
MOD_RE = r"\[[^\]]*\]"


def _bases(col):
    """DECOY_-stripped peptidoforms, as deeplc_finetune.base_pf."""
    return pc.replace_substring(col, "DECOY_", "")


def uniq(lib, prefix, n):
    t0 = time.time()
    pf = pq.ParquetFile(lib)
    parts = []
    for rg in range(pf.num_row_groups):
        col = _bases(pf.read_row_group(rg, columns=["peptidoform"]).column(0))
        u = pc.unique(col)
        if isinstance(u, pa.ChunkedArray):
            u = u.combine_chunks()
        # large_string for the concatenation: the per-row-group uniques of a 200M-row
        # table add up past the 2 GiB `string` offset limit ("offset overflow while
        # concatenating arrays"); the sequences go to the shards as large_string too.
        parts.append(pc.cast(u, pa.large_string()))
    allu = pc.unique(pa.concat_arrays(parts))
    n_raw = len(allu)
    stripped = pc.replace_substring_regex(allu, MOD_RE, "")
    allu = pc.filter(allu, pc.match_substring_regex(stripped, STD_RE))
    n_u = len(allu)
    per = -(-n_u // n)
    for k in range(n):
        sl = allu.slice(k * per, per)
        pq.write_table(pa.table({"seq": sl}), f"{prefix}{k:02d}.parquet", compression="snappy")
    print(f"uniq: {pf.metadata.num_rows} rows -> {n_raw} unique bases -> {n_u} standard, "
          f"{n} shards of <= {per} -> {prefix}NN.parquet in {time.time() - t0:.0f}s", flush=True)


def predict(a):
    import deeplc_finetune as W  # imports deeplc before numpy/torch, as the worker must
    import deeplc
    import torch

    args = argparse.Namespace(seed_path=a.seed, q_train=0.01, window_holdout_frac=a.holdout,
                              max_ref=0, multihead=a.heads)
    ref = W.build_reference(args)
    cal = W.fit_multihead(args, ref)
    torch.set_num_threads(max(1, a.threads))
    seqs = pq.read_table(a.shard).column("seq").to_pylist()
    if a.limit:
        seqs = seqs[: a.limit]
    ref_t = W.ref_psms_for_transform(cal, args)
    out = np.full(len(seqs), np.nan, dtype=np.float32)
    chunk = 100_000
    t0 = time.time()
    print(f"predicting {len(seqs)} sequences from {a.shard} (torch threads={torch.get_num_threads()})", flush=True)
    with W.quiet_deeplc_progress():
        for s in range(0, len(seqs), chunk):
            t1 = time.time()
            batch = seqs[s:s + chunk]
            p = W.agg(deeplc.predict_and_calibrate(batch, psm_list_reference=ref_t, calibration=cal))
            if len(p) != len(batch):
                raise SystemExit(f"DeepLC returned {len(p)} predictions for {len(batch)} sequences")
            out[s:s + len(batch)] = np.asarray(p, dtype=np.float32)
            done = s + len(batch)
            rate = len(batch) / max(1e-9, time.time() - t1)
            print(f"  {done}/{len(seqs)}  {rate:.0f}/s  ETA {(len(seqs) - done) / rate / 60:.1f} min", flush=True)
    pq.write_table(pa.table({"seq": pa.array(seqs, pa.string()), "irt": pa.array(out, pa.float32())}),
                   a.out, compression="snappy")
    print(f"wrote {len(seqs)} predictions -> {a.out} in {time.time() - t0:.0f}s", flush=True)


def merge(lib, out, preds):
    import pandas as pd

    t0 = time.time()
    t = pa.concat_tables([pq.read_table(p) for p in preds])
    seq = t.column("seq").to_numpy(zero_copy_only=False)
    irt = t.column("irt").to_numpy().astype(np.float32)
    idx = pd.Index(seq)
    if not idx.is_unique:
        raise SystemExit("prediction shards overlap: a sequence was predicted twice")
    print(f"merge: {len(idx)} predicted sequences indexed in {time.time() - t0:.0f}s", flush=True)
    pf = pq.ParquetFile(lib)
    w = None
    rows = re = 0
    for rg in range(pf.num_row_groups):
        rt = pf.read_row_group(rg)
        bases = _bases(rt.column("peptidoform")).to_numpy(zero_copy_only=False)
        pos = idx.get_indexer(bases)
        orig = rt.column("predicted_irt").to_numpy().astype(np.float32)
        new = orig.copy()
        hit = np.flatnonzero(pos >= 0)
        vals = irt[pos[hit]]
        fin = np.isfinite(vals)
        new[hit[fin]] = vals[fin]
        re += int(fin.sum())
        rows += len(bases)
        rt = rt.set_column(rt.schema.get_field_index("predicted_irt"), "predicted_irt", pa.array(new, pa.float32()))
        if w is None:
            w = pq.ParquetWriter(out, rt.schema, compression="snappy")
        w.write_table(rt)
    w.close()
    summary = {"rows": rows, "repredicted": re, "retained_imported": rows - re,
               "model": "the base model calibrated over multiple heads (sharded)", "lib_in": lib, "lib_out": out}
    with open(out + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"merge: rows={rows} repredicted={re} retained_imported={rows - re} -> {out} in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    u = sub.add_parser("uniq"); u.add_argument("lib"); u.add_argument("prefix"); u.add_argument("n", type=int)
    p = sub.add_parser("predict"); p.add_argument("shard"); p.add_argument("seed"); p.add_argument("out")
    p.add_argument("--holdout", type=float, default=0.0); p.add_argument("--threads", type=int, default=16)
    p.add_argument("--heads", type=int, default=80); p.add_argument("--limit", type=int, default=0)
    m = sub.add_parser("merge"); m.add_argument("lib"); m.add_argument("out"); m.add_argument("preds", nargs="+")
    a = ap.parse_args()
    if a.mode == "uniq":
        uniq(a.lib, a.prefix, a.n)
    elif a.mode == "predict":
        predict(a)
    else:
        merge(a.lib, a.out, a.preds)


if __name__ == "__main__":
    main()
