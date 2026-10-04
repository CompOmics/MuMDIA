"""MBR re-extraction tier (`mbr.reextract`, run-experiment, diaPASEF): docs/TIMS_QUANT_ROADMAP.md 4k, 4l.

A precursor confident in >= min_anchor_runs OTHER runs, not confident in this run but
extracted here, is retraced at its cross-run expected RT and at the anchor runs' 1/K0, and
accepted when its evidence there beats a same-trace RT-shift null.

  mbr_reextract.py prep  <scored> <psms_csv> <centroid_csv> <out_dir> [--q-anchor --min-anchor-runs]
  mbr_reextract.py score <scored> <frag_csv> <out_dir> <out_scored> [--q-anchor --q-transfer --seed --rescuable]

<scored> is the MBR-off scored table: every precursor confident in >= min_anchor_runs other runs and not
confident here is a target, also those the rescuable tier would take, because re-extracted values quantify
better (docs/TIMS_QUANT_ROADMAP.md 4l). With --rescuable, score then adds mbr_worker.py's transfers for the
rows it did not accept.

prep writes, per run i, <out_dir>/r<i>/:
  targets.parquet                candidate_id, expected_rt, expected_im, n_anchor
  psms.parquet                   candidate_id, peak_rank (0), apex_rt, apex_im: retrace's apex
  chromatograms.centroid.parquet extract's centroid traces of the targets only
The expected RT is mbr_worker.py's (binned-median maps through run 0, median over the anchor
runs); the expected 1/K0 is the median of the anchor runs' apex_im, each moved onto this run's
scale by a per-run median offset over the precursors confident in both runs.

The engine then retraces each run's targets into <out_dir>/r<i>/chromatograms.parquet.

score reads those traces. Evidence at the expected RT: the cosine of square-rooted fragment
areas (+-4 scans) against the anchor runs' L1-normalised pass-1 areas (<frag_csv>, quant's
fragment tables), the median Pearson of each fragment with the sum of the others (+-6
scans), and the log10 summed area. Null: the same three values in the same traces at K
random positions >= 15 s from the expected RT (same candidate, same run, no ion there). The
score is a logistic regression of target against null, cross-fitted in 2 folds by candidate,
so no row is scored by a model that saw it. Transfer q = (null >= s + 1) / K / (targets >= s),
running minimum, as in mbr_worker.py. An accepted row gets apex_rt = the expected RT, its PSM
q columns lowered to the transfer q, is_transferred and transfer_q; <out_dir>/r<i>/accepted.parquet
lists every transfer of the run with re-extracted traces, for quant to read them from those traces.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from _lib_io import write_engine_parquet
from mbr_worker import binned_map

K, H_AREA, H_COEL, MIN_OFF_S, MAX_GRID_S = 10, 4, 6, 15.0, 1.5


def read_scored(path, q_anchor):
    """Per run: target rows (candidate_id, apex_rt, confident, done)."""
    cols = ["candidate_id", "source", "label", "apex_rt", "q_value"]
    has_tr = "is_transferred" in pq.read_schema(path).names  # scored_combined when the rescuable tier took none
    t = pq.read_table(path, columns=cols + ["is_transferred"] * has_tr, filters=[("label", "=", "target")]).to_pandas()
    if not has_tr:
        t["is_transferred"] = False
    t["confident"] = (t.q_value <= q_anchor) & ~t.is_transferred
    t["done"] = (t.q_value <= q_anchor) | t.is_transferred
    return {int(s): g.drop_duplicates("candidate_id") for s, g in t.groupby("source")}


def anchors(sc):
    return {i: dict(zip(g.candidate_id[g.confident], g.apex_rt[g.confident])) for i, g in sc.items()}


def prep(a):
    psms, cents = a.inputs.split(","), a.centroid.split(",")
    n = len(psms)
    sc = read_scored(a.scored, a.q_anchor)
    anc = anchors(sc)

    def m(x, y):
        sh = [c for c in anc[x] if c in anc[y]]
        return binned_map([anc[x][c] for c in sh], [anc[y][c] for c in sh]) if x != y and len(sh) >= 200 else (lambda q: q)
    to_ref = {i: m(i, 0) for i in range(n)}
    from_ref = {i: m(0, i) for i in range(n)}
    im = {}
    for i, p in enumerate(psms):
        e = pq.read_table(p, columns=["candidate_id", "peak_rank", "apex_im"]).to_pandas()
        im[i] = e[e.peak_rank == 0].drop_duplicates("candidate_id").set_index("candidate_id").apex_im
    off = {}
    for i in range(n):
        sh = list(set(anc[i]) & set(anc[0]))
        off[i] = float(np.nanmedian(im[i].reindex(sh).values - im[0].reindex(sh).values)) if i and sh else 0.0
    allc = set().union(*[set(x) for x in anc.values()])
    for i in range(n):
        g = sc.get(i, pd.DataFrame(columns=["candidate_id", "done"]))
        done, ext = set(g.candidate_id[g.done]), set(g.candidate_id)
        rows = []
        for c in sorted(allc):
            if c in done or c not in ext:
                continue
            js = [j for j in range(n) if j != i and c in anc[j]]
            if len(js) < a.min_anchor_runs:
                continue
            rt = float(from_ref[i](np.median([to_ref[j](anc[j][c]) for j in js])))
            ims = [im[j].get(c, np.nan) - off[j] for j in js]
            rows.append((c, rt, float(np.nanmedian(ims)) + off[i] if np.isfinite(ims).any() else np.nan, len(js)))
        t = pd.DataFrame(rows, columns=["candidate_id", "expected_rt", "expected_im", "n_anchor"])
        o = f"{a.out_dir}/r{i}"
        os.makedirs(o, exist_ok=True)
        cid = pa.array(t.candidate_id.values, pa.uint32())
        pq.write_table(pa.table({"candidate_id": cid, "expected_rt": t.expected_rt.values,
                                 "expected_im": t.expected_im.values, "n_anchor": pa.array(t.n_anchor.values, pa.int32())}),
                       f"{o}/targets.parquet", compression="snappy")
        pq.write_table(pa.table({"candidate_id": cid, "peak_rank": pa.array(np.zeros(len(t), np.int32)),
                                 "apex_rt": t.expected_rt.values, "apex_im": t.expected_im.values}),
                       f"{o}/psms.parquet", compression="snappy")
        ch = ds.dataset(cents[i]).to_table(filter=ds.field("candidate_id").isin(cid))
        pq.write_table(ch, f"{o}/chromatograms.centroid.parquet", compression="snappy", row_group_size=131072)
        print(f"  run {i}: {len(t)} re-extraction targets, 1/K0 offset to run 0 {off[i]:+.4f}", flush=True)


def evidence(F, j, cons):
    """(cosine, co-elution, log10 area) at grid index j."""
    A = F[:, j - H_AREA:j + H_AREA + 1].sum(1)
    sa, sc = np.sqrt(A), np.sqrt(cons)
    cos = float(sa @ sc / (np.linalg.norm(sa) * np.linalg.norm(sc))) if A.any() else 0.0
    X = F[:, j - H_COEL:j + H_COEL + 1]
    tot = X.sum(0)
    r = [np.corrcoef(x, tot - x)[0, 1] for x in X if x.std() > 0 and (tot - x).std() > 0]
    return cos, float(np.median(r)) if r else 0.0, float(np.log10(A.sum() + 1.0))


def score_run(args):
    i, path, want, cons, seed = args  # want: cid -> expected rt; cons: cid -> {fragment: share}
    rng = np.random.default_rng([seed, i])
    tr = {}
    flt = ds.field("candidate_id").isin(pa.array(list(want), pa.uint32()))
    for b in ds.dataset(path).to_batches(columns=["candidate_id", "frag_name", "rt", "intensity"], filter=flt):
        cid, fn = b.column("candidate_id").to_numpy(), b.column("frag_name").to_pylist()
        rts, its = b.column("rt"), b.column("intensity")
        ro, io = rts.offsets.to_numpy(), its.offsets.to_numpy()
        rv, iv = rts.values.to_numpy(), its.values.to_numpy()
        for k in range(len(cid)):
            if fn[k].startswith("ms1_") or ro[k + 1] == ro[k]:
                continue
            tr.setdefault(int(cid[k]), []).append((fn[k], rv[ro[k]:ro[k + 1]], iv[io[k]:io[k + 1]]))
    out = []  # (run, cid, rt, cos, coel, larea, null)
    for c, e in want.items():
        rows = [r for r in tr.get(c, []) if r[0] in cons.get(c, {})]
        if len(rows) < 2:
            continue
        rt = rows[0][1].astype(np.float64)
        rows = [r for r in rows if len(r[1]) == len(rt)]
        F = np.array([r[2] for r in rows], np.float64)
        cv = np.array([cons[c][r[0]] for r in rows])
        j = int(np.argmin(np.abs(rt - e)))
        if j < H_COEL or j >= len(rt) - H_COEL or abs(rt[j] - e) > MAX_GRID_S:
            continue
        out.append((i, c, e, *evidence(F, j, cv), False))
        ks = [k for k in range(H_COEL, len(rt) - H_COEL) if abs(rt[k] - e) >= MIN_OFF_S]
        for k in (rng.choice(ks, K) if ks else []):
            out.append((i, c, float(rt[k]), *evidence(F, k, cv), True))
    return out


def logistic(X, y, w, iters=25, ridge=1e-6):
    """Weighted logistic regression by Newton steps on standardised columns; returns a scorer."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ b, -30, 30)))
        g = Z.T @ (w * (p - y)) + ridge * b
        Hm = (Z * (w * p * (1 - p))[:, None]).T @ Z + ridge * np.eye(len(b))
        step = np.linalg.solve(Hm, g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda X2: np.c_[np.ones(len(X2)), (X2 - mu) / sd] @ b


def transfer_q(t, null):
    """q per target score: (null >= s + 1) / K / (targets >= s), running minimum."""
    o = np.argsort(-t)
    ts, ns = t[o], np.sort(null)[::-1]
    fdr = (np.searchsorted(-ns, -ts, side="right") + 1) / K / np.arange(1, len(ts) + 1)
    q = np.minimum.accumulate(fdr[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = q
    return out


def score(a):
    frags = a.inputs.split(",")
    n = len(frags)
    sc = read_scored(a.scored, a.q_anchor)
    anc = anchors(sc)
    want = {}
    for i in range(n):
        t = pq.read_table(f"{a.out_dir}/r{i}/targets.parquet").to_pandas()
        want[i] = dict(zip(t.candidate_id.astype(int), t.expected_rt))
    need = set().union(*[set(w) for w in want.values()])
    parts = []
    for j, fp in enumerate(frags):
        f = pq.read_table(fp, columns=["candidate_id", "fragment_name", "quantity"]).to_pandas()
        f = f[f.candidate_id.isin(need) & f.candidate_id.isin(set(anc[j])) & (f.quantity > 0)]
        f["s"] = f.quantity / f.groupby("candidate_id").quantity.transform("sum")
        parts.append(f[["candidate_id", "fragment_name", "s"]])
    cons = {}
    for (c, fn), v in pd.concat(parts).groupby(["candidate_id", "fragment_name"]).s.sum().items():
        cons.setdefault(int(c), {})[fn] = v
    jobs = [(i, f"{a.out_dir}/r{i}/chromatograms.parquet", want[i], {c: cons[c] for c in want[i] if c in cons}, a.seed)
            for i in range(n) if want[i] and os.path.exists(f"{a.out_dir}/r{i}/chromatograms.parquet")]
    with Pool(max(1, min(len(jobs), a.processes))) as pool:
        res = pool.map(score_run, jobs)
    x = pd.DataFrame([r for rr in res for r in rr], columns=["run", "candidate_id", "rt", "cos", "coel", "larea", "null"])
    acc = pd.DataFrame(columns=["run", "candidate_id", "rt", "transfer_q"])
    if len(x) and (~x.null).any() and x.null.any():
        F = x[["cos", "coel", "larea"]].to_numpy()
        y = (~x.null).to_numpy().astype(float)
        fold = x.candidate_id.to_numpy() % 2
        s = np.empty(len(x))
        for f in (0, 1):
            tr, te = fold != f, fold == f
            w = np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))  # balanced classes
            s[te] = logistic(F[tr], y[tr], w)(F[te])
        tgt = ~x.null.to_numpy()
        q = transfer_q(s[tgt], s[~tgt])
        T = x[tgt].assign(transfer_q=q)
        acc = T[T.transfer_q <= a.q_transfer]
        thr = s[tgt][q <= a.q_transfer].min() if (q <= a.q_transfer).any() else np.inf
        print(f"MBR re-extraction: {int(tgt.sum())} targets tested, {len(acc)} accepted at q <= {a.q_transfer}; "
              f"RT-shift null: {int((~tgt).sum())} draws, {int((s[~tgt] >= thr).sum())} at or above the threshold "
              f"(expected false {(s[~tgt] >= thr).sum() / K:.0f})")
    else:
        print("MBR re-extraction: nothing to test")
    acc = acc[["run", "candidate_id", "rt", "transfer_q"]]
    n_rx = len(acc)
    if a.rescuable:
        # The rescuable tier's transfers (mbr_worker.py) for the rows re-extraction did not accept. They keep
        # their own apex (rt NaN) and their transfer q; the two tiers were tested against separate nulls.
        r = pq.read_table(a.rescuable, columns=["candidate_id", "source", "label", "is_transferred", "transfer_q"],
                          filters=[("is_transferred", "=", True)]).to_pandas()
        r = r[r.label == "target"].drop_duplicates(["candidate_id", "source"])
        r = pd.DataFrame({"run": r.source.astype(int).values, "candidate_id": r.candidate_id.astype(int).values,
                          "rt": np.nan, "transfer_q": r.transfer_q.values})
        took = set(zip(acc.run, acc.candidate_id))
        r = r[[k not in took for k in zip(r.run, r.candidate_id)]]
        acc = pd.concat([acc, r], ignore_index=True)
        print(f"  rescuable transfers added for rows re-extraction did not accept: {len(r)}")
    for i in range(n):
        ai = acc[acc.run == i]
        # Quant reads these from the re-extracted traces: every transfer of this run that has them.
        own = np.sort(ai.candidate_id[ai.candidate_id.isin(set(want[i]))].values).astype(np.uint32)
        pq.write_table(pa.table({"candidate_id": pa.array(own, pa.uint32())}),
                       f"{a.out_dir}/r{i}/accepted.parquet", compression="snappy")
        print(f"  run {i}: +{int(ai.rt.notna().sum())} re-extracted, +{int(ai.rt.isna().sum())} rescuable transfers")
    full = pq.read_table(a.scored).to_pandas()
    if "is_transferred" not in full.columns:
        full["is_transferred"] = False
        full["transfer_q"] = np.nan
    key = lambda c, s_: (np.asarray(s_).astype(np.int64) << 32) | np.asarray(c).astype(np.int64)
    k = pd.Series(np.arange(len(acc)), index=key(acc.candidate_id.values, acc.run.values))
    pos = k.reindex(key(full.candidate_id.values, full.source.values)).to_numpy()
    hit = np.isfinite(pos) & (full.label == "target").to_numpy()
    pi = pos[hit].astype(int)
    tq = acc.transfer_q.to_numpy()[pi]
    rt = acc.rt.to_numpy()[pi]
    full.loc[hit, "apex_rt"] = np.where(np.isfinite(rt), rt, full.loc[hit, "apex_rt"].to_numpy())
    for col in ("q_value", "run_psm_q", "experiment_psm_q"):
        if col in full.columns:
            full.loc[hit, col] = np.minimum(full.loc[hit, col].to_numpy(dtype=float), tq)
    full.loc[hit, "is_transferred"] = True
    full.loc[hit, "transfer_q"] = tq
    write_engine_parquet(full, a.out_scored)
    print(f"wrote {a.out_scored} ({int(hit.sum())} transferred rows, {n_rx} of them re-extracted)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "score"])
    ap.add_argument("scored")
    ap.add_argument("inputs", help="prep: per-run psms (apex_im), score: per-run pass-1 fragment tables; ',' joined")
    ap.add_argument("rest", nargs="+", help="prep: <centroid_csv> <out_dir>; score: <out_dir> <out_scored>")
    ap.add_argument("--q-anchor", type=float, default=0.01)
    ap.add_argument("--min-anchor-runs", type=int, default=2)
    ap.add_argument("--q-transfer", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--rescuable", default=None,
                    help="score: mbr_worker.py's augmented scored table; its transfers are added for the rows "
                         "re-extraction did not accept (mbr.rescuable)")
    a = ap.parse_args()
    if a.mode == "prep":
        a.centroid, a.out_dir = a.rest
        prep(a)
    else:
        a.out_dir, a.out_scored = a.rest
        score(a)


if __name__ == "__main__":
    sys.exit(main())
