"""Assemble a per-run library from `mumdia prescan` survivors.

DEPRECATED: `mumdia sub-library` does the same thing in the engine, streaming both tables
instead of holding the precursor table and every label string in memory (44 GB on the
142.7M-precursor 8-12-mer library, against about 1.6 GB). Prefer:

    mumdia sub-library --lib-precursors P --lib-fragments F --survivors S         --out-precursors OP --out-fragments OF

This script stays for recipes that already call it and is unchanged otherwise.


Usage: python assemble_survivors.py <lib_precursors> <lib_fragments> <prescan_survivors>
                                    <out_precursors> <out_fragments>

Keeps every precursor whose `peptidoform_id` pair (target and its decoy) has at least one surviving member and
drops the rest, then renumbers `candidate_id` to the contiguous, precursor-m/z-ordered range
the fragment index requires (`index.rs`), remapping the fragment rows to match. The
fragment table is streamed one parquet row group at a time; only the precursor table is
held in memory.

Pair-linked by construction: a target and its decoy share `peptidoform_id` (the decoy
builders copy the target row), so the pair gets one decision; charge states are screened
separately, each on its own precursor m/z window. The screen is symmetric already (docs/21), but a scrambled decoy's tag set is not
its target's, and taking the union over the group removes the last asymmetry: no member of
a pair can be kept while the other is dropped, so target/decoy exchangeability downstream
is exactly what it was in the full library.
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from _lib_io import narrow_table


def main():
    lib_prec, lib_frag, survivors, out_prec, out_frag = sys.argv[1:6]
    prec = pq.read_table(lib_prec)
    n = prec.num_rows
    cid = prec.column("candidate_id").to_numpy().astype(np.int64)
    if not np.array_equal(cid, np.arange(n)):
        raise SystemExit("library precursors must carry candidate_id 0..n-1 in row order")
    # The pair key: a decoy row is a copy of its target's row, so the two share
    # `peptidoform_id` (and `base_peptide_id`, which also spans the charge states and
    # would union those in too; measured on a 9-mer immunopeptidomics library, that turned
    # 23% survivors into 60% kept). `peptidoform_id` links exactly the pair.
    bpid = prec.column("peptidoform_id").to_numpy().astype(np.int64)
    label = np.asarray(prec.column("label").to_pylist())

    surv = pq.read_table(survivors, columns=["candidate_id"]).column(0).to_numpy().astype(np.int64)
    if len(surv) and (surv.min() < 0 or surv.max() >= n):
        raise SystemExit("survivor candidate_id outside the library's range")
    survived = np.zeros(n, dtype=bool)
    survived[surv] = True

    # Union over the pair: the target and its decoy travel together.
    keep_group = np.zeros(int(bpid.max()) + 1, dtype=bool)
    keep_group[bpid[survived]] = True
    keep = keep_group[bpid]
    n_keep = int(keep.sum())
    new_id = np.full(n, -1, dtype=np.int64)
    new_id[keep] = np.arange(n_keep, dtype=np.int64)

    t_all = int((label == "target").sum())
    d_all = n - t_all
    t_keep = int(((label == "target") & keep).sum())
    d_keep = n_keep - t_keep
    print(
        f"survivors {int(survived.sum())} of {n}; pair-linked union keeps {n_keep} precursors "
        f"({n_keep / max(n, 1):.1%}): targets {t_keep}/{t_all}, decoys {d_keep}/{d_all}, "
        f"target:decoy {t_keep / max(d_keep, 1):.4f}",
        flush=True,
    )

    kept = prec.filter(pa.array(keep))
    cols = {name: kept.column(name) for name in kept.schema.names}
    cols["candidate_id"] = pa.array(new_id[keep].astype(np.uint32), pa.uint32())
    out = pa.table(cols)
    pq.write_table(narrow_table(out), out_prec, compression="snappy")

    pf = pq.ParquetFile(lib_frag)
    writer = None
    n_rows = 0
    try:
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            old = t.column("candidate_id").to_numpy().astype(np.int64)
            sel = new_id[old] >= 0
            if not sel.any():
                continue
            t = t.filter(pa.array(sel))
            mapped = pa.array(new_id[old[sel]].astype(np.uint32), pa.uint32())
            t = t.set_column(t.schema.get_field_index("candidate_id"), "candidate_id", mapped)
            t = narrow_table(t)
            if writer is None:
                writer = pq.ParquetWriter(out_frag, t.schema, compression="snappy")
            writer.write_table(t)
            n_rows += t.num_rows
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote {n_keep} precursors and {n_rows} fragment rows -> {out_prec}, {out_frag}", flush=True)


if __name__ == "__main__":
    main()
